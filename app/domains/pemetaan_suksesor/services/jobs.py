import uuid
from typing import Dict, Any, Optional
from datetime import datetime

class JobService:
    _instance = None
    _jobs: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobService, cls).__new__(cls)
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
            "error": None
        }
        return job_id

    def update_job(self, job_id: str, status: str, result: Any = None, error: str = None):
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            if status in ["completed", "failed"]:
                self._jobs[job_id]["finished_at"] = datetime.now().isoformat()
            if result:
                self._jobs[job_id]["result"] = result
            if error:
                self._jobs[job_id]["error"] = error

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

def get_job_service() -> JobService:
    return JobService()
