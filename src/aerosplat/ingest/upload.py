"""Uploaded captures: streamed to a staging directory in the API request, ingested by a CPU job.

    captures/.upload-<job id>/        staging, one per upload (= per ingest job)
    ├── files/                        what the browser sent, as is
    ├── upload.json                   written last, atomically: "the upload is complete"
    └── capture/                      the capture tree being built (job), then renamed

The API (aerosplat.server.api_v2) only streams the files to disk, does the cheap checks
and seals the staging directory, then queues a `cpu` job. The job (`run_ingest_job`,
in a process of the CPU worker) hashes the images, lays the capture tree out *inside
the staging directory*, renames it into `captures/<id>/` in one step and registers it
in the catalog in one transaction. The catalog is what makes a capture exist, so:

* a capture is never visible half-done: until the catalog commit it is not listed, and
  until the rename its directory is not even in place;
* killed at any point (cancel, crash, power) the job leaves either the staging
  directory (re-running moves the half-built tree back and starts over) or a
  complete, renamed capture directory with a marker file holding its description
  (re-running registers it). Re-running a finished job is a no-op;
* nothing outside the staging directory and `captures/<id>/` is touched, and a
  `captures/<id>/` that this upload did not create is never modified.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

from aerosplat.ingest.images import IMAGE_EXTS
from aerosplat.jobs.models import EventType, IngestSpec, Job, JobKind, JobStatus
from aerosplat.jobs.queue import JobNotFoundError, JobQueue
from aerosplat.store.catalog import Catalog, CatalogError, NotFoundError
from aerosplat.store.models import Capture, CaptureSource, Image, Site, utcnow
from aerosplat.store.paths import CaptureDirs

STAGING_PREFIX = ".upload-"
FILES = "files"
SEAL = "upload.json"
BUILD = "capture"
# In the capture directory: the full description until it is registered (so a re-run can
# register it), then just {"job", "registered": true} (so a re-run knows it is done).
MARKER = ".ingest.json"

KEEP_FINISHED_S = 3600.0         # staging of a finished job: kept an hour for diagnosis
KEEP_ORPHAN_S = 6 * 3600.0       # staging without a job (API died mid-upload)
PROGRESS_EVERY_S = 1.0

STAGES = (("check", "Перевірка завантаження"), ("hash", "Хешування й розкладка кадрів"),
          ("register", "Реєстрація в каталозі"))


class IngestError(RuntimeError):
    """The upload cannot become a capture; the message says why (shown on the job)."""


def staging_name(job_id: str) -> str:
    return f"{STAGING_PREFIX}{job_id}"


def staging_dir(catalog: Catalog, site: Site, name: str) -> Path:
    if not name.startswith(STAGING_PREFIX) or "/" in name or "\\" in name or ".." in name:
        raise IngestError(f"bad staging name '{name}'")
    return catalog.site_dirs(site).captures / name


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def seal(staging: Path, spec: IngestSpec) -> None:
    """Mark the staging directory complete. Must happen before the job is queued."""
    _write_json_atomic(staging / SEAL, {"spec": spec.model_dump(), "sealed_at": utcnow().isoformat()})


def is_sealed(staging: Path) -> bool:
    return (staging / SEAL).is_file()


# ── housekeeping (called by the API on each upload) ──────────────────────
def _newest_mtime(d: Path) -> float:
    newest = d.stat().st_mtime
    for p in d.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


def sweep_stale_uploads(catalog: Catalog, site: Site, q: JobQueue, *,
                        now: Optional[float] = None) -> list[str]:
    """Remove staging directories nobody will use any more; returns their names.

    Kept: those of queued/running jobs, of jobs finished less than an hour ago, and those
    still being written (no job yet, touched in the last six hours)."""
    root = catalog.site_dirs(site).captures
    if not root.is_dir():
        return []
    now = time.time() if now is None else now
    removed = []
    for d in sorted(root.glob(f"{STAGING_PREFIX}*")):
        if not d.is_dir():
            continue
        job_id = d.name[len(STAGING_PREFIX):]
        try:
            job: Optional[Job] = q.get(job_id)
        except JobNotFoundError:
            job = None
        if job is None:
            stale = now - _newest_mtime(d) > KEEP_ORPHAN_S
        elif job.status.terminal:
            done = job.finished_at.timestamp() if job.finished_at else 0.0
            stale = now - done > KEEP_FINISHED_S
        else:
            stale = False
        if stale:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
    return removed


def active_ingest(q: JobQueue, site: Site, capture_id: str) -> Optional[Job]:
    """A queued/running ingest of this capture id, if any."""
    for job in q.list_jobs(statuses=[JobStatus.queued, JobStatus.running], site_key=site.key,
                           limit=500):
        if job.kind == JobKind.ingest and job.payload.get("capture_id") == capture_id:
            return job
    return None


# ── the job ──────────────────────────────────────────────────────────────
Emit = Callable[[EventType, dict[str, Any]], None]


class _Stages:
    def __init__(self, emit: Emit):
        self.emit = emit
        self.t0 = 0.0
        self.current: Optional[tuple[int, str, str]] = None

    def start(self, name: str) -> None:
        idx = next(i for i, (n, _) in enumerate(STAGES, 1) if n == name)
        label = dict(STAGES)[name]
        self.current = (idx, name, label)
        self.t0 = time.perf_counter()
        self._send("started")
        print(f"▶  {label}", flush=True)

    def progress(self, done: int, total: int) -> None:
        self._send("started", progress={"done": done, "total": total})

    def done(self, **extra: Any) -> None:
        self._send("done", seconds=round(time.perf_counter() - self.t0, 2), **extra)

    def skipped(self, name: str, reason: str) -> None:
        self.start(name)
        self._send("skipped", reason=reason)

    def failed(self, error: str) -> None:
        if self.current is not None:
            self._send("failed", seconds=round(time.perf_counter() - self.t0, 2), error=error)

    def _send(self, state: str, **extra: Any) -> None:
        assert self.current is not None
        idx, name, label = self.current
        self.emit(EventType.stage, {"stage": name, "label": label, "state": state, "index": idx,
                                    "total": len(STAGES), **extra})


def _restore_build(staging: Path) -> None:
    """Undo a half-built capture tree of an interrupted attempt: files go back to files/."""
    build = staging / BUILD
    if not build.exists():
        return
    files = staging / FILES
    files.mkdir(exist_ok=True)
    for sub in ("images", "source"):
        d = build / sub
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and not (files / p.name).exists():
                    os.replace(p, files / p.name)
    shutil.rmtree(build)


def _read_marker(capture_root: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads((capture_root / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _prepare(spec: IngestSpec, staging: Path, on_progress: Callable[[int, int], None]
             ) -> tuple[Capture, list[Image]]:
    files, dest = staging / FILES, CaptureDirs(staging / BUILD)
    source = CaptureSource(spec.source)
    if source == CaptureSource.earthstudio:
        from aerosplat.ingest.earthstudio import ATTRIBUTION_FILE, prepare_earthstudio

        jsons = sorted(p for p in files.iterdir() if p.suffix.lower() == ".json")
        if len(jsons) != 1:
            raise IngestError(f"Earth Studio upload needs exactly one export .json (got {len(jsons)})")
        extras = [p for p in files.iterdir() if p.name == ATTRIBUTION_FILE or p.suffix == ".esp"]
        return prepare_earthstudio(jsons[0], files, dest, capture_id=spec.capture_id,
                                   extra_files=extras, author=spec.author, move=True,
                                   on_progress=on_progress)
    from aerosplat.ingest.folder import prepare_folder

    return prepare_folder(files, dest, capture_id=spec.capture_id, source=source,
                          license=spec.license, author=spec.author, notes=spec.notes, move=True,
                          on_progress=on_progress)


def run_ingest(catalog: Catalog, spec: IngestSpec, job_id: str, emit: Emit) -> dict[str, Any]:
    """Turn a sealed upload into a registered capture. Idempotent; see the module docstring.

    Returns the job result ({"site", "capture", "images", "bytes"}); raises IngestError
    (or an adapter's FileNotFoundError / ValueError) when the upload cannot be ingested."""
    st = _Stages(emit)
    try:
        site = catalog.find_site(spec.site)
    except (NotFoundError, CatalogError) as e:
        raise IngestError(str(e)) from None
    staging = staging_dir(catalog, site, spec.staging)
    final = catalog.site_dirs(site).capture(spec.capture_id)

    def result(images: list[Image]) -> dict[str, Any]:
        return {"site": site.key, "capture": spec.capture_id, "images": len(images),
                "bytes": sum(i.bytes for i in images)}

    try:
        st.start("check")
        marker = _read_marker(final.root)
        ours = marker is not None and marker.get("job") == job_id
        registered = spec.capture_id in {c.id for c in catalog.list_captures(site)}
        if registered:
            if not ours:
                raise IngestError(f"capture '{spec.capture_id}' already exists in {site.key}")
            # Registered by an earlier attempt of this very job (finished, or died while
            # cleaning up): nothing left to do but the cleanup.
            images = catalog.list_images(site, spec.capture_id)
            _write_json_atomic(final.root / MARKER, {"job": job_id, "registered": True})
            shutil.rmtree(staging, ignore_errors=True)
            st.done(note="already registered by this job")
            for name, _ in STAGES[1:]:
                st.skipped(name, "already done")
            return result(images)
        if final.root.exists() and not ours:
            raise IngestError(f"directory {final.root} exists but is not a registered capture; "
                              f"it was not created by this upload, so it is left alone — "
                              f"move it away or choose another capture id")
        if not ours:
            if not staging.is_dir() or not is_sealed(staging):
                raise IngestError(f"upload {spec.staging} is missing or incomplete "
                                  f"(the request did not finish); upload again")
            names = [p.name for p in (staging / FILES).iterdir()] if (staging / FILES).is_dir() else []
            if not any(Path(n).suffix.lower() in IMAGE_EXTS for n in names) \
                    and not (staging / BUILD).exists():
                raise IngestError("the upload contains no images (JPEG/PNG/TIFF)")
        st.done()

        if ours:
            # Renamed into place by an earlier attempt, not registered yet: finish that.
            if "capture" not in marker or "images" not in marker:
                raise IngestError(f"{final.root / MARKER} says registered, the catalog disagrees; "
                                  f"check the catalog before re-running")
            capture = Capture(**marker["capture"])
            images = [Image(**i) for i in marker["images"]]
            st.skipped("hash", "done by an earlier attempt")
        else:
            st.start("hash")
            _restore_build(staging)
            last = [0.0]

            def on_progress(done: int, total: int) -> None:
                now = time.monotonic()
                if done == total or now - last[0] >= PROGRESS_EVERY_S:
                    last[0] = now
                    st.progress(done, total)
                    print(f"   {done}/{total} кадрів", flush=True)

            capture, images = _prepare(spec, staging, on_progress)
            build = staging / BUILD
            _write_json_atomic(build / MARKER, {
                "job": job_id, "staging": spec.staging,
                "capture": capture.model_dump(mode="json"),
                "images": [i.model_dump(mode="json") for i in images]})
            st.done(images=len(images))
            st.start("register")
            os.rename(build, final.root)            # atomic on one volume; fails if it exists
            print(f"   → {final.root}", flush=True)
        if ours:
            st.start("register")
        catalog.add_capture(site, capture, images)  # one transaction: the capture now exists
        _write_json_atomic(final.root / MARKER, {"job": job_id, "registered": True})
        shutil.rmtree(staging, ignore_errors=True)
        st.done()
        return result(images)
    except (Exception, SystemExit) as e:     # find_footage reports a mismatch as SystemExit
        st.failed(f"{type(e).__name__}: {e}")
        raise


def run_ingest_job(q: JobQueue, job: Job, catalog: Catalog) -> int:
    """Job-process entry point (aerosplat.jobs.child): 0 = succeeded."""
    spec = IngestSpec(**job.payload)
    print(f"📥 {spec.site} · {spec.capture_id} · {spec.files} файлів, "
          f"{spec.bytes / 1e6:.1f} MB", flush=True)
    try:
        res = run_ingest(catalog, spec, job.id, lambda t, d: q.add_event(job.id, t, d))
    except (IngestError, FileNotFoundError, FileExistsError, ValueError, CatalogError,
            SystemExit) as e:
        print(f"❌ {type(e).__name__}: {e}", flush=True)
        return 1
    q.merge_result(job.id, res)
    print(f"✅ захоплення {res['capture']}: {res['images']} кадрів", flush=True)
    return 0

