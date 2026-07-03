import asyncio
import json
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket
from datetime import datetime


class JobService:
    _instance = None
    _jobs: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobService, cls).__new__(cls)
            cls._instance._ws_connections: Dict[str, list[WebSocket]] = {}
        return cls._instance

    def create_job(self, task_name: str) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "id": job_id,
            "task": task_name,
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
        return job_id

    def update_job(self, job_id: str, status: str, result: Any = None, error: str = None):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            if status in ["completed", "failed"]:
                self._jobs[job_id]["finished_at"] = datetime.now().isoformat()
            if result is not None:
                self._jobs[job_id]["result"] = result
            if error is not None:
                self._jobs[job_id]["error"] = error

            # Notify WebSocket listeners
            self._broadcast(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    # ── WebSocket management ──────────────────────────────────────────

    async def connect(self, job_id: str, ws: WebSocket):
        """Accept and register a WebSocket connection for a job."""
        await ws.accept()
        if job_id not in self._ws_connections:
            self._ws_connections[job_id] = []
        self._ws_connections[job_id].append(ws)

        # Immediately send current job state so the client doesn't have to poll first
        job = self._jobs.get(job_id)
        if job:
            await self._send_to_ws(ws, job)

    def disconnect(self, job_id: str, ws: WebSocket):
        """Remove a WebSocket connection."""
        if job_id in self._ws_connections:
            try:
                self._ws_connections[job_id].remove(ws)
            except ValueError:
                pass
            if not self._ws_connections[job_id]:
                del self._ws_connections[job_id]

    def _broadcast(self, job_id: str):
        """Push updated job state to all WebSocket listeners for this job.

        Runs fire-and-forget — does not block the caller (which is usually
        running inside a BackgroundTask, not the async event loop).
        """
        job = self._jobs.get(job_id)
        if not job:
            return
        connections = self._ws_connections.get(job_id, [])
        if not connections:
            return
        for ws in list(connections):
            # Schedule the send on the running event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_to_ws(ws, job))
            except RuntimeError:
                # No running loop — skip (e.g., called from sync context)
                pass

    @staticmethod
    async def _send_to_ws(ws: WebSocket, data: Any):
        """Safely send JSON data to a WebSocket, removing broken connections."""
        try:
            await ws.send_json(data)
        except Exception:
            # Connection already closed on client side — will be cleaned
            # up on next disconnect() call
            pass


def get_job_service() -> JobService:
    return JobService()