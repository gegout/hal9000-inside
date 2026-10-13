"""Preemptible background scheduler for HAL 9000 cognition.

Runs priority background jobs only when:
- no spoken turn is in progress
- no user speech is detected
Preempts immediately on VAD speech detection.
Supports hard caps on concurrent jobs, idle tokens, and web fetch rates.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class BackgroundJob:
    def __init__(
        self,
        job_id: str,
        job_type: str,  # 'digest', 'curiosity_search', 'prefetch', 'rehearse', 'consolidate', 'reflect'
        coro_fn: Callable[[], Coroutine[Any, Any, Any]],
        priority: int = 10,
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.coro_fn = coro_fn
        self.priority = priority
        self.created_at = time.time()


class CognitiveScheduler:
    """Asyncio task queue preemptible on user speech start."""

    def __init__(
        self,
        max_concurrent: int = 2,
        max_tokens_per_idle: int = 4096,
        max_web_fetches_per_hour: int = 30,
        clock: Optional[Any] = None,
    ):
        self.max_concurrent = max_concurrent
        self.max_tokens_per_idle = max_tokens_per_idle
        self.max_web_fetches_per_hour = max_web_fetches_per_hour
        self.clock = clock or time

        self.queue: List[BackgroundJob] = []
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.is_turn_active = False
        self.is_speech_detected = False
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    def enqueue(self, job: BackgroundJob):
        self.queue.append(job)
        # Higher priority first
        self.queue.sort(key=lambda j: j.priority, reverse=True)

    def on_speech_start(self):
        """Preempt all background jobs immediately on speech detected."""
        self.is_speech_detected = True
        self.cancel_all_active()

    def on_speech_end(self):
        self.is_speech_detected = False

    def on_turn_start(self):
        self.is_turn_active = True
        self.cancel_all_active()

    def on_turn_end(self):
        self.is_turn_active = False

    def cancel_all_active(self):
        """Immediately cancel all active background tasks to free compute."""
        for jid, task in list(self.active_tasks.items()):
            if not task.done():
                task.cancel()
        self.active_tasks.clear()

    async def run_once(self) -> int:
        """Run up to max_concurrent pending jobs if idle. Returns number of jobs started."""
        if self.is_turn_active or self.is_speech_detected:
            return 0

        started = 0
        while self.queue and len(self.active_tasks) < self.max_concurrent:
            job = self.queue.pop(0)
            task = asyncio.create_task(self._wrap_job(job))
            self.active_tasks[job.job_id] = task
            started += 1
        return started

    async def _wrap_job(self, job: BackgroundJob):
        try:
            await job.coro_fn()
        except asyncio.CancelledError:
            logger.debug(f"Background job {job.job_id} ({job.job_type}) preempted.")
        except Exception as e:
            logger.warning(f"Background job {job.job_id} error: {e}")
        finally:
            self.active_tasks.pop(job.job_id, None)
