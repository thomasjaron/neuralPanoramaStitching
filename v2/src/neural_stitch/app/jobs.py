"""Jobs for the web app.

Two worker threads: one for alignment checks, one for stitching. A new check never waits behind a
long neural refinement, and at most one refinement runs at a time.
"""
import json
import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..config import MatchConfig, StitchConfig, match_preset
from ..images import InputImage, decode_image, save_png, thumbnail_jpeg
from ..pipeline import Cancelled, CheckResult, ProgressEvent, Stitcher

log = logging.getLogger(__name__)


class State(StrEnum):
    CHECKING = "checking"
    CHECKED = "checked"  # images align; waiting for "start"
    NOT_ALIGNABLE = "not_alignable"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


IDLE = {State.CHECKED, State.NOT_ALIGNABLE, State.DONE, State.FAILED, State.CANCELLED}


class JobError(Exception):
    """A request that cannot be served (bad input, wrong state). Shown to the user."""


@dataclass
class Job:
    id: str
    dir: Path
    images: list[InputImage]
    state: State = State.CHECKING
    created: float = field(default_factory=time.time)
    check: CheckResult | None = None
    error: str | None = None
    files: dict[str, str] = field(default_factory=dict)  # role -> file name
    metrics: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    cond: threading.Condition = field(default_factory=threading.Condition)
    preview_version: int = 0

    def emit(self, type_: str, **data) -> None:
        with self.cond:
            self.events.append({"seq": len(self.events) + 1, "type": type_, "time": time.time(), **data})
            self.cond.notify_all()

    def set_state(self, state: State, **data) -> None:
        self.state = state
        self.emit("state", state=state.value, **data)

    def events_after(self, seq: int) -> list[dict]:
        with self.cond:
            return self.events[seq:]

    def public(self) -> dict:
        return {
            "id": self.id,
            "state": self.state.value,
            "error": self.error,
            "images": [{"index": k, "name": im.name, "width": im.size[0], "height": im.size[1],
                        "thumbnail": f"thumb_{k}.jpg"} for k, im in enumerate(self.images)],
            "report": self.check.report.model_dump() if self.check else None,
            "files": self.files,
            "metrics": {k: v for k, v in self.metrics.items() if k != "neural_history"},
            "history": self.metrics.get("neural_history", []),
        }


class JobManager:
    def __init__(self, stitcher: Stitcher, root: Path, max_images: int = 30,
                 max_image_mb: int = 50, ttl_s: int = 6 * 3600):
        self.stitcher = stitcher
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_images = max_images
        self.max_image_bytes = max_image_mb * 1024 * 1024
        self.ttl_s = ttl_s
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.check_queue: queue.Queue = queue.Queue()
        self.stitch_queue: queue.Queue = queue.Queue()
        for name, q in (("check-worker", self.check_queue), ("stitch-worker", self.stitch_queue)):
            threading.Thread(target=self._work, args=(q,), name=name, daemon=True).start()

    # --- requests --------------------------------------------------------------------------------

    def create(self, uploads: list[tuple[str, bytes]], speed: str = "accurate") -> Job:
        if len(uploads) < 2:
            raise JobError("Please upload at least 2 images.")
        if len(uploads) > self.max_images:
            raise JobError(f"Please upload at most {self.max_images} images.")
        self._cleanup()
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True)
        images = []
        try:
            for name, data in uploads:
                if len(data) > self.max_image_bytes:
                    raise JobError(f"{name} is larger than {self.max_image_bytes // 2**20} MB.")
                try:
                    img = decode_image(data, Path(name).stem or name, self.stitcher.config.max_input_side)
                except Exception as e:
                    raise JobError(f"{name} is not a readable image.") from e
                (job_dir / f"thumb_{len(images)}.jpg").write_bytes(thumbnail_jpeg(img.rgb))
                images.append(img)
        except JobError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        job = Job(id=job_id, dir=job_dir, images=images)
        with self.lock:
            self.jobs[job_id] = job
        job.set_state(State.CHECKING)
        self.check_queue.put((job, self._check, match_preset(self.stitcher.config.match, speed)))
        return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def start(self, job: Job, config: StitchConfig) -> None:
        if job.state not in (State.CHECKED, State.DONE, State.CANCELLED, State.FAILED) or job.check is None \
                or not job.check.report.ok:
            raise JobError(f"Cannot start a job in state '{job.state.value}'.")
        job.cancel.clear()
        job.error = None
        job.set_state(State.QUEUED)
        self.stitch_queue.put((job, self._stitch, config))

    def cancel(self, job: Job) -> None:
        job.cancel.set()

    def delete(self, job: Job) -> None:
        job.cancel.set()
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.dir, ignore_errors=True)

    def _cleanup(self) -> None:
        now = time.time()
        with self.lock:
            old = [j for j in self.jobs.values() if j.state in IDLE and now - j.created > self.ttl_s]
        for j in old:
            self.delete(j)

    # --- worker ----------------------------------------------------------------------------------

    def _work(self, tasks: queue.Queue) -> None:
        while True:
            job, run, arg = tasks.get()
            if job.id not in self.jobs:
                continue
            try:
                run(job, arg)
            except Cancelled:
                job.set_state(State.CANCELLED)
            except Exception as e:  # report everything to the user, keep the worker alive
                log.exception("job %s failed", job.id)
                job.error = str(e) or type(e).__name__
                job.set_state(State.FAILED, error=job.error)

    def _progress(self, job: Job):
        def on_progress(e: ProgressEvent) -> None:
            data = {"stage": e.stage, "message": e.message, "fraction": e.fraction,
                    "data": {k: v for k, v in e.data.items() if isinstance(v, (int, float, str, type(None)))}}
            if e.preview is not None:
                job.preview_version += 1
                save_png(e.preview, job.dir / "preview.png")
                job.files["preview"] = "preview.png"
                data["preview"] = f"preview.png?v={job.preview_version}"
            job.emit("progress", **data)
        return on_progress

    def _check(self, job: Job, match: MatchConfig) -> None:
        check = self.stitcher.check(job.images, self._progress(job), job.cancel.is_set, match=match)
        job.check = check
        job.metrics = dict(check.metrics)
        if check.preview is not None:
            save_png(check.preview, job.dir / "check_preview.png")
            job.files["check_preview"] = "check_preview.png"
        job.set_state(State.CHECKED if check.report.ok else State.NOT_ALIGNABLE,
                      report=check.report.model_dump())

    def _stitch(self, job: Job, config: StitchConfig) -> None:
        job.set_state(State.RUNNING)
        res = self.stitcher.stitch(job.images, job.check, config, self._progress(job), job.cancel.is_set)
        save_png(res.panorama, job.dir / "panorama.png")
        save_png(res.classic, job.dir / "panorama_classic.png")
        homs = {job.images[k].name: H.tolist() for k, H in res.homographies.items()}
        (job.dir / "homographies.json").write_text(json.dumps(homs, indent=2))
        job.files |= {"panorama": "panorama.png", "classic": "panorama_classic.png",
                      "homographies": "homographies.json"}
        job.metrics = res.metrics
        job.set_state(State.DONE)
