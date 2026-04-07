"""In-memory background job manager for long-running inference tasks.

Follows the same patterns as ChatSessionManager: UUID-keyed, thread-safe,
TTL-pruned. Spawns daemon threads so the SageMaker container process can
exit cleanly even if jobs are still running.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)


class JobManager:
    """Thread-safe in-memory store for background inference jobs."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """Spawn a daemon thread to run *fn* and return a job ID immediately."""
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "status": "running",
                "result": None,
                "error": None,
                "created_at": time.monotonic(),
            }
        thread = threading.Thread(
            target=self._run_job, args=(job_id, fn, *args), kwargs=kwargs, daemon=True,
        )
        thread.start()
        self._cleanup_expired()
        return job_id

    def poll(self, job_id: str) -> dict[str, Any]:
        """Return the current state of a job."""
        self._cleanup_expired()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return {"job_id": job_id, "status": "not_found"}
            response: dict[str, Any] = {"job_id": job_id, "status": job["status"]}
            if job["status"] == "completed":
                response["result"] = job["result"]
            elif job["status"] == "failed":
                response["error"] = job["error"]
            return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_job(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Thread target that executes *fn* and stores the outcome."""
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = "completed"
                    self._jobs[job_id]["result"] = result
        except Exception as exc:
            logger.exception("Background job %s failed", job_id)
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(exc)

    def _cleanup_expired(self) -> None:
        """Remove completed/failed jobs older than *ttl_seconds*."""
        now = time.monotonic()
        with self._lock:
            expired = [
                jid
                for jid, j in self._jobs.items()
                if j["status"] != "running" and (now - j["created_at"]) > self._ttl_seconds
            ]
            for jid in expired:
                del self._jobs[jid]
