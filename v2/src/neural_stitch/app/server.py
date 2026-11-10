"""FastAPI app: upload -> alignment check -> start -> live progress (SSE) -> results."""
import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import RESOLUTIONS, Resolution
from ..pipeline import Stitcher
from .jobs import Job, JobError, JobManager

STATIC = Path(__file__).parent / "static"
FILE_NAME = re.compile(r"^[\w.-]+\.(png|jpg|json)$")


class StartOptions(BaseModel):
    neural: bool = True
    max_iters: int | None = Field(None, ge=1, le=100_000)
    resolution: Resolution | None = None  # "native" or a preset like "1080p"
    max_side: int | None = Field(None, ge=64, le=50_000, description="Long-side limit for 'native'")


def create_app(stitcher: Stitcher | None = None, jobs_dir: Path | None = None,
               max_images: int = 30) -> FastAPI:
    stitcher = stitcher or Stitcher()
    manager = JobManager(stitcher, jobs_dir or Path(tempfile.gettempdir()) / "neural-stitch-jobs",
                         max_images=max_images)
    app = FastAPI(title="neural-stitch")
    app.state.manager = manager

    def job_or_404(job_id: str) -> Job:
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found.") from None

    @app.get("/api/info")
    def info() -> dict:
        cfg = stitcher.config
        return {"device": str(stitcher.device), "backend": cfg.match.backend, "max_images": max_images,
                "neural": cfg.neural.model_dump(include={"max_iters", "min_iters", "enabled"}),
                "output": cfg.output.model_dump(),
                "resolutions": [{"name": k, "short_side": n, "label": label} for k, (n, label) in RESOLUTIONS.items()]}

    @app.post("/api/jobs")
    async def create_job(files: Annotated[list[UploadFile], File()],
                         speed: Annotated[Literal["accurate", "fast"], Form()] = "accurate") -> dict:
        uploads = [(f.filename or f"image_{k}", await f.read()) for k, f in enumerate(files)]
        try:
            job = await asyncio.to_thread(manager.create, uploads, speed)
        except JobError as e:
            raise HTTPException(400, str(e)) from None
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return job_or_404(job_id).public()

    @app.post("/api/jobs/{job_id}/start")
    def start_job(job_id: str, options: StartOptions) -> dict:
        job = job_or_404(job_id)
        cfg = stitcher.config.model_copy(deep=True)
        cfg.neural.enabled = options.neural
        if options.max_iters:
            cfg.neural.max_iters = options.max_iters
            cfg.neural.min_iters = min(cfg.neural.min_iters, options.max_iters)
        if options.resolution:
            cfg.output.resolution = options.resolution
        if options.max_side:
            cfg.output.max_side = options.max_side
        try:
            manager.start(job, cfg)
        except JobError as e:
            raise HTTPException(409, str(e)) from None
        return job.public()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        job = job_or_404(job_id)
        manager.cancel(job)
        return job.public()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        manager.delete(job_or_404(job_id))
        return {"deleted": job_id}

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request) -> StreamingResponse:
        job = job_or_404(job_id)
        last = int(request.headers.get("last-event-id") or 0)

        async def stream():
            nonlocal last
            idle = 0.0
            while not await request.is_disconnected():
                new = job.events_after(last)
                for e in new:
                    last = e["seq"]
                    yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps(e)}\n\n"
                idle = 0.0 if new else idle + 0.25
                if idle >= 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/files/{name}")
    def get_file(job_id: str, name: str, download: bool = False) -> FileResponse:
        job = job_or_404(job_id)
        path = job.dir / name
        if not FILE_NAME.match(name) or not path.is_file():
            raise HTTPException(404, "File not found.")
        return FileResponse(path, filename=name if download else None,
                            headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
