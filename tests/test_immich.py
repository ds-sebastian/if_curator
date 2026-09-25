import threading
import time
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from if_curator.immich import Immich, ImmichError, bounded_map


class Response:
    def __init__(self, body=None, status=200, content=b""):
        self.body, self.status_code, self.content = body, status, content

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(self.status_code)


def client(monkeypatch, handler):
    immich = Immich("http://immich:2283/api/", "key")
    calls = []

    def request(method, url, timeout, **kwargs):
        calls.append((method, url.removeprefix("http://immich:2283/api"), kwargs))
        return handler(method, url.removeprefix("http://immich:2283/api"), kwargs)

    monkeypatch.setattr(immich.session, "request", request)
    return immich, calls


def test_people_pages_and_skips_unnamed(monkeypatch):
    pages = {1: [{"id": "a", "name": "Ann"}, {"id": "b", "name": ""}], 2: [{"id": "c", "name": "Cy"}]}
    immich, calls = client(
        monkeypatch,
        lambda m, path, kw: Response({"people": pages[kw["params"]["page"]], "hasNextPage": kw["params"]["page"] < 2}),
    )
    assert [p["id"] for p in immich.people()] == ["a", "c"]
    assert immich.session.headers["x-api-key"] == "key"


def test_photos_pages_with_filters(monkeypatch):
    def handler(method, path, kw):
        page = kw["json"]["page"]
        return Response({"assets": {"items": [{"id": f"x{page}"}], "nextPage": "2" if page == 1 else None}})

    immich, calls = client(monkeypatch, handler)
    assert [a["id"] for a in immich.photos("person", years=2)] == ["x1", "x2"]
    query = calls[0][2]["json"]
    assert (calls[0][0], calls[0][1]) == ("POST", "/search/metadata")
    assert query["personIds"] == ["person"] and query["type"] == "IMAGE" and query["withPeople"]
    assert query["takenAfter"][:4].isdigit()


def test_target_faces_falls_back_to_face_lookup(monkeypatch):
    box = dict(boundingBoxX1=1, boundingBoxY1=2, boundingBoxX2=3, boundingBoxY2=4, imageWidth=10, imageHeight=10)
    faces = [{"id": "f1", "person": {"id": "me"}, **box}, {"id": "f2", "person": {"id": "you"}, **box}]
    immich, calls = client(monkeypatch, lambda m, path, kw: Response(faces))
    nested = {"id": "a", "people": [{"id": "me", "faces": [{"id": "f0", **box}]}]}
    assert [f["id"] for f in immich.target_faces(nested, "me")] == ["f0"] and not calls
    assert [f["id"] for f in immich.target_faces({"id": "a", "people": [{"id": "me"}]}, "me")] == ["f1"]
    assert calls[0][1:] == ("/faces", {"params": {"id": "a"}})


def test_image_is_upright_bgr(monkeypatch):
    rgb = np.zeros((20, 40, 3), np.uint8)
    rgb[:, :, 0] = 255  # red
    buffer = BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° on display
    Image.fromarray(rgb).save(buffer, "JPEG", exif=exif)
    immich, calls = client(monkeypatch, lambda m, path, kw: Response(content=buffer.getvalue()))
    image = immich.image("id", original=True)
    assert image.shape == (40, 20, 3) and image[..., 2].mean() > 200 and image[..., 0].mean() < 50
    assert calls[0][1] == "/assets/id/original"
    immich.image("id")
    assert calls[1][1:] == ("/assets/id/thumbnail", {"params": {"size": "preview"}})


def test_bad_key_is_explained(monkeypatch):
    immich, _ = client(monkeypatch, lambda m, path, kw: Response(status=401))
    with pytest.raises(ImmichError, match="API key"):
        immich.people()


def test_bounded_map_keeps_order_and_bounds_work():
    running, peak, lock = 0, 0, threading.Lock()

    def work(i):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.001 * (i % 3))
        with lock:
            running -= 1
        return i * i

    assert list(bounded_map(work, range(50), workers=4)) == [i * i for i in range(50)]
    assert peak <= 4
