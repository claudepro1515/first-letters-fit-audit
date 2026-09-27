"""Minimal, dependency-light reader for the Vesuvius Challenge open-data OME-Zarr (v2) stores.

Reads arbitrary boxes of any pyramid level straight from the public S3 bucket over HTTPS,
with parallel chunk fetches and an optional on-disk chunk cache. Needs only numpy, requests
and numcodecs (no zarr / fsspec / s3fs), so it runs on any laptop, Windows included.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from numcodecs import get_codec

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

_session_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=3)
        s.mount("https://", adapter)
        _session_local.s = s
    return s


def http_get(url: str, timeout: float = 60.0) -> bytes | None:
    """GET a URL; returns None on 404 (missing zarr chunk == fill value)."""
    for attempt in range(4):
        try:
            r = _session().get(url, timeout=timeout)
        except requests.RequestException:
            if attempt == 3:
                raise
            continue
        if r.status_code == 404 or r.status_code == 403:
            return None
        if r.status_code == 200:
            return r.content
        if attempt == 3:
            r.raise_for_status()
    return None


def get_json(url: str) -> dict | None:
    b = http_get(url)
    return None if b is None else json.loads(b)


def list_prefixes(prefix: str) -> list[str]:
    """List 'sub-folders' of a bucket prefix (S3 ListObjectsV2 with delimiter)."""
    import re

    out, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&delimiter=/&prefix={prefix}"
        if token:
            url += "&continuation-token=" + requests.utils.quote(token, safe="")
        xml = http_get(url).decode()
        out += [p for p in re.findall(r"<Prefix>([^<]*)</Prefix>", xml) if p != prefix]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
        if not m:
            return out
        token = m.group(1)


def list_keys(prefix: str) -> list[tuple[str, int]]:
    import re

    xml = http_get(f"{BUCKET}/?list-type=2&prefix={prefix}").decode()
    return [(k, int(s)) for k, s in re.findall(r"<Key>([^<]*)</Key>.*?<Size>([^<]*)</Size>", xml, re.S)]


class ZArray:
    """One zarr v2 array (one pyramid level) on HTTP."""

    def __init__(self, url: str, cache_dir: str | None = None, workers: int = 32):
        self.url = url.rstrip("/")
        meta = get_json(self.url + "/.zarray")
        if meta is None:
            raise FileNotFoundError(self.url + "/.zarray")
        self.meta = meta
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta.get("fill_value") or 0
        self.sep = meta.get("dimension_separator", ".")
        self.order = meta.get("order", "C")
        comp = meta.get("compressor")
        self.codec = get_codec(comp) if comp else None
        self.filters = [get_codec(f) for f in (meta.get("filters") or [])]
        self.cache_dir = cache_dir
        self.workers = workers

    def _chunk_key(self, idx) -> str:
        return self.sep.join(str(i) for i in idx)

    def _load_chunk(self, idx) -> np.ndarray | None:
        key = self._chunk_key(idx)
        raw = None
        cpath = None
        if self.cache_dir:
            safe = self.url.replace("https://", "").replace("/", "_")
            cpath = os.path.join(self.cache_dir, safe, key.replace("/", "_"))
            if os.path.exists(cpath):
                with open(cpath, "rb") as f:
                    raw = f.read()
                if raw == b"":
                    return None
        if raw is None:
            raw = http_get(self.url + "/" + key)
            if cpath:
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                tmp = f"{cpath}.{os.getpid()}.{threading.get_ident()}.tmp"  # unique: several processes may share a cache
                with open(tmp, "wb") as f:
                    f.write(raw or b"")
                os.replace(tmp, cpath)
            if raw is None:
                return None
        buf = self.codec.decode(raw) if self.codec else raw
        for flt in reversed(self.filters):
            buf = flt.decode(buf)
        arr = np.frombuffer(buf, dtype=self.dtype)
        return arr.reshape(self.chunks, order=self.order)

    def read(self, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        """Read the box [z0:z1, y0:y1, x0:x1] (clipped to the array)."""
        lo = [max(0, v) for v in (z0, y0, x0)]
        hi = [min(s, v) for s, v in zip(self.shape, (z1, y1, x1))]
        out = np.full([max(0, h - l) for l, h in zip(lo, hi)], self.fill, dtype=self.dtype)
        if out.size == 0:
            return out
        ranges = [range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, self.chunks)]
        idxs = [(a, b, c) for a in ranges[0] for b in ranges[1] for c in ranges[2]]

        def job(idx):
            return idx, self._load_chunk(idx)

        with ThreadPoolExecutor(self.workers) as ex:
            for idx, chunk in ex.map(job, idxs):
                if chunk is None:
                    continue
                sl_out, sl_chunk = [], []
                for d in range(3):
                    c0 = idx[d] * self.chunks[d]
                    a = max(lo[d], c0)
                    b = min(hi[d], c0 + self.chunks[d])
                    sl_out.append(slice(a - lo[d], b - lo[d]))
                    sl_chunk.append(slice(a - c0, b - c0))
                out[tuple(sl_out)] = chunk[tuple(sl_chunk)]
        return out


class OmeZarr:
    """An OME-Zarr multiscale group; levels opened lazily."""

    def __init__(self, url: str, cache_dir: str | None = None):
        self.url = url.rstrip("/")
        self.attrs = get_json(self.url + "/.zattrs") or {}
        self.cache_dir = cache_dir
        self._levels: dict[str, ZArray] = {}
        self.scales: dict[str, float] = {}
        for ms in self.attrs.get("multiscales", []):
            for ds in ms.get("datasets", []):
                sc = 1.0
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        sc = float(t["scale"][-1])
                self.scales[str(ds["path"])] = sc

    def level(self, path) -> ZArray:
        path = str(path)
        if path not in self._levels:
            self._levels[path] = ZArray(f"{self.url}/{path}", cache_dir=self.cache_dir)
        return self._levels[path]
