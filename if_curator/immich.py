"""The few Immich API calls this tool needs."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from io import BytesIO
from itertools import islice

import numpy as np
import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class ImmichError(RuntimeError):
    pass


BOX_KEYS = ("boundingBoxX1", "boundingBoxY1", "boundingBoxX2", "boundingBoxY2", "imageWidth", "imageHeight")
# A photo that can't be fetched or decoded is skipped; auth errors (ImmichError) are not.
DOWNLOAD_ERRORS = (requests.RequestException, OSError, Image.DecompressionBombError)


class Immich:
    def __init__(self, url: str, api_key: str):
        self.url = url.rstrip("/").removesuffix("/api") + "/api"
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})
        # Searches are read-only, so retrying POST is safe.
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=(502, 503, 504), allowed_methods=None)
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _request(self, method: str, path: str, timeout: float = 30, **kwargs) -> requests.Response:
        response = self.session.request(method, self.url + path, timeout=timeout, **kwargs)
        if response.status_code in (401, 403):
            raise ImmichError("Immich rejected the API key. It needs access to people, search, faces and assets.")
        response.raise_for_status()
        return response

    def people(self) -> list[dict]:
        """Named, visible people."""
        people, page = [], 1
        while True:
            data = self._request("GET", "/people", params={"page": page, "size": 1000}).json()
            people += data["people"]
            if not data.get("hasNextPage"):
                return [p for p in people if p.get("name")]
            page += 1

    def photos(self, person_id: str, years: int) -> list[dict]:
        """Photos of a person taken in the last `years` years, with face boxes."""
        taken_after = datetime.now(UTC) - timedelta(days=round(365.25 * years))
        query = {
            "personIds": [person_id],
            "type": "IMAGE",
            "takenAfter": taken_after.isoformat(),
            "withPeople": True,
            "size": 1000,
        }
        assets, page = [], 1
        while page:
            data = self._request("POST", "/search/metadata", json={**query, "page": page}).json()["assets"]
            assets += data["items"]
            page = int(data["nextPage"]) if data.get("nextPage") else None
        return assets

    def target_faces(self, asset: dict, person_id: str) -> list[dict]:
        """The person's face boxes in an asset, looked up separately if the search omitted them."""
        faces = [f for p in asset.get("people") or [] if p.get("id") == person_id for f in p.get("faces") or []]
        if faces and all(key in face for face in faces for key in BOX_KEYS):
            return faces
        response = self._request("GET", "/faces", params={"id": asset["id"]})
        return [f for f in response.json() if (f.get("person") or {}).get("id") == person_id]

    def image(self, asset_id: str, original: bool = False) -> np.ndarray:
        """A decoded, upright BGR image: the original, or Immich's preview."""
        if original:
            response = self._request("GET", f"/assets/{asset_id}/original", timeout=120)
        else:
            response = self._request("GET", f"/assets/{asset_id}/thumbnail", params={"size": "preview"})
        with Image.open(BytesIO(response.content)) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])


def bounded_map(fn, items, workers: int = 8):
    """Like executor.map, but only a few results are held in memory at once."""
    items = iter(items)
    with ThreadPoolExecutor(workers) as pool:
        pending = deque(pool.submit(fn, item) for item in islice(items, workers * 2))
        while pending:
            result = pending.popleft().result()
            pending.extend(pool.submit(fn, item) for item in islice(items, 1))
            yield result
