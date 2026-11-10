import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from neural_stitch.app.server import create_app
from neural_stitch.pipeline import Stitcher


def png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def wait_for(client: TestClient, job_id: str, states: set[str], timeout: float = 120) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job stuck in {job['state']}")


@pytest.fixture
def client(fast_config, tmp_path):
    with TestClient(create_app(Stitcher(fast_config), tmp_path, max_images=5)) as c:
        yield c


def test_full_flow(client, scene):
    images, _ = scene
    files = [("files", (f"{im.name}.png", png(im.rgb), "image/png")) for im in images]
    r = client.post("/api/jobs", files=files)
    assert r.status_code == 200, r.text
    job = r.json()
    assert len(job["images"]) == 4

    job = wait_for(client, job["id"], {"checked", "not_alignable", "failed"})
    assert job["state"] == "checked", job
    assert [im["used"] for im in job["report"]["images"]] == [True, False, True, True]
    assert client.get(f"/api/jobs/{job['id']}/files/thumb_0.jpg").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/files/{job['files']['check_preview']}").status_code == 200

    r = client.post(f"/api/jobs/{job['id']}/start", json={"neural": True, "max_iters": 40, "resolution": "144p"})
    assert r.status_code == 200, r.text
    job = wait_for(client, job["id"], {"done", "failed"})
    assert job["state"] == "done", job
    assert job["metrics"]["neural_iterations"] <= 40
    pano = Image.open(io.BytesIO(client.get(f"/api/jobs/{job['id']}/files/panorama.png").content))
    assert min(pano.size) == 144  # 144p = short side
    assert job["metrics"]["output_resolution"] == "144p"
    assert job["metrics"]["output_size"] == f"{pano.size[0]} × {pano.size[1]}"
    for role in ("panorama", "classic", "homographies"):
        assert client.get(f"/api/jobs/{job['id']}/files/{job['files'][role]}").status_code == 200
    dl = client.get(f"/api/jobs/{job['id']}/files/panorama.png?download=true")
    assert "attachment" in dl.headers["content-disposition"]

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_rejects_bad_uploads(client, scene):
    images, _ = scene
    one = [("files", ("a.png", png(images[0].rgb), "image/png"))]
    assert client.post("/api/jobs", files=one).status_code == 400
    bad = one + [("files", ("notes.txt", b"hello", "text/plain"))]
    r = client.post("/api/jobs", files=bad)
    assert r.status_code == 400 and "notes.txt" in r.json()["detail"]
    too_many = one * 6
    assert client.post("/api/jobs", files=too_many).status_code == 400


def test_unknown_job_and_file(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/start", json={}).status_code == 404


def test_not_alignable(client, scene):
    from .conftest import texture
    files = [("files", (f"{k}.png", png(texture(50 + k, 200, 150)), "image/png")) for k in range(2)]
    job = client.post("/api/jobs", files=files).json()
    job = wait_for(client, job["id"], {"checked", "not_alignable", "failed"})
    assert job["state"] == "not_alignable"
    assert client.post(f"/api/jobs/{job['id']}/start", json={}).status_code == 409


def test_rejects_unknown_resolution(client):
    assert client.post("/api/jobs/nope/start", json={"resolution": "999p"}).status_code == 422


def test_static_frontend(client):
    r = client.get("/")
    assert r.status_code == 200 and "neural-stitch" in r.text
    info = client.get("/api/info").json()
    assert info["backend"] == "sift"
    assert [r["name"] for r in info["resolutions"]][:2] == ["144p", "240p"]


def test_check_does_not_wait_for_a_running_refinement(fast_config, tmp_path, scene):
    fast_config.neural.min_iters = 100_000  # the refinement only ends when cancelled
    with TestClient(create_app(Stitcher(fast_config), tmp_path)) as client:
        _check_runs_alongside_refinement(client, scene)


def _check_runs_alongside_refinement(client, scene):
    images, _ = scene
    files = [("files", (f"{im.name}.png", png(im.rgb), "image/png")) for im in images]
    first = client.post("/api/jobs", files=files).json()
    wait_for(client, first["id"], {"checked"})
    client.post(f"/api/jobs/{first['id']}/start", json={"max_iters": 100_000})
    wait_for(client, first["id"], {"running"})

    second = client.post("/api/jobs", files=files, data={"speed": "fast"}).json()
    second = wait_for(client, second["id"], {"checked", "not_alignable", "failed"}, timeout=60)
    assert second["state"] == "checked"
    assert client.get(f"/api/jobs/{first['id']}").json()["state"] == "running"
    client.post(f"/api/jobs/{first['id']}/cancel")
    assert wait_for(client, first["id"], {"cancelled", "done", "failed"})["state"] == "cancelled"


def test_rejects_unknown_speed(client, scene):
    images, _ = scene
    files = [("files", (f"{im.name}.png", png(im.rgb), "image/png")) for im in images[:2]]
    assert client.post("/api/jobs", files=files, data={"speed": "turbo"}).status_code == 422
