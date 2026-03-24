"""Unit tests for the background job manager."""

from __future__ import annotations

import threading
import time

import pytest

from handlers.job_manager import JobManager


class TestSubmit:
    """Tests for JobManager.submit."""

    def test_submit_returns_uuid(self) -> None:
        """submit() returns a UUID-formatted string."""
        jm = JobManager()
        job_id = jm.submit(lambda: "ok")
        # UUID4 format: 8-4-4-4-12
        assert len(job_id.split("-")) == 5

    def test_poll_running_job(self) -> None:
        """A job that hasn't finished yet shows status 'running'."""
        barrier = threading.Event()
        jm = JobManager()
        job_id = jm.submit(lambda: barrier.wait(timeout=5))
        result = jm.poll(job_id)
        assert result["status"] == "running"
        assert result["job_id"] == job_id
        barrier.set()  # unblock the thread

    def test_poll_completed_job(self) -> None:
        """A finished job returns status 'completed' with the result."""
        jm = JobManager()
        job_id = jm.submit(lambda: {"findings": {"ascites": 0.87}})
        # Wait for completion
        for _ in range(50):
            result = jm.poll(job_id)
            if result["status"] == "completed":
                break
            time.sleep(0.05)
        assert result["status"] == "completed"
        assert result["result"] == {"findings": {"ascites": 0.87}}

    def test_poll_failed_job(self) -> None:
        """A job that raises an exception shows status 'failed' with the error message."""
        def bad_fn() -> None:
            raise RuntimeError("CUDA out of memory")

        jm = JobManager()
        job_id = jm.submit(bad_fn)
        for _ in range(50):
            result = jm.poll(job_id)
            if result["status"] == "failed":
                break
            time.sleep(0.05)
        assert result["status"] == "failed"
        assert "CUDA out of memory" in result["error"]

    def test_poll_unknown_job(self) -> None:
        """Polling a non-existent job returns 'not_found'."""
        jm = JobManager()
        result = jm.poll("no-such-id")
        assert result["status"] == "not_found"


class TestTTLCleanup:
    """Tests for TTL-based job cleanup."""

    def test_ttl_cleanup_removes_old(self) -> None:
        """Completed jobs older than TTL are cleaned up."""
        jm = JobManager(ttl_seconds=0)  # expire immediately
        job_id = jm.submit(lambda: "done")
        # Wait for completion
        for _ in range(50):
            if jm.poll(job_id)["status"] == "completed":
                break
            time.sleep(0.05)
        # Next poll triggers cleanup — stale completed job is removed
        time.sleep(0.01)
        result = jm.poll(job_id)
        assert result["status"] == "not_found"

    def test_ttl_cleanup_preserves_running(self) -> None:
        """Running jobs are never cleaned up regardless of TTL."""
        barrier = threading.Event()
        jm = JobManager(ttl_seconds=0)
        job_id = jm.submit(lambda: barrier.wait(timeout=5))
        time.sleep(0.05)
        # Trigger cleanup
        result = jm.poll(job_id)
        assert result["status"] == "running"
        barrier.set()


class TestConcurrency:
    """Tests for thread safety."""

    def test_concurrent_submits(self) -> None:
        """Multiple threads can submit and poll without errors."""
        jm = JobManager()
        job_ids: list[str] = []
        errors: list[Exception] = []

        def submit_and_poll() -> None:
            try:
                jid = jm.submit(lambda: "ok")
                job_ids.append(jid)
                jm.poll(jid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit_and_poll) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(job_ids) == 20
        # All job IDs are unique
        assert len(set(job_ids)) == 20
