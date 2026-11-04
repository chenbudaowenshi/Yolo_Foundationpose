#!/usr/bin/env python3
"""A bounded single-worker queue for non-blocking pose registration."""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AsyncResult:
    request: Any
    value: Any
    error: str | None
    duration_s: float


class LatestPoseRegistrationWorker:
    """Run one expensive registration at a time without blocking video I/O.

    Requests are deliberately not accumulated. A stale backlog is dangerous
    for robotics because each result describes an old object location.
    """

    def __init__(self, function: Callable[[Any], Any], *, name: str = "pose-registration") -> None:
        self.function = function
        self._requests: queue.Queue[Any | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[AsyncResult] = queue.Queue()
        self._busy = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._requests.empty()

    def submit(self, request: Any) -> bool:
        if self._closed or self.busy:
            return False
        try:
            self._requests.put_nowait(request)
            return True
        except queue.Full:
            return False

    def poll(self) -> list[AsyncResult]:
        results: list[AsyncResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            self._busy.set()
            started = time.perf_counter()
            value = None
            error = None
            try:
                value = self.function(request)
            except Exception:
                error = traceback.format_exc()
            finally:
                duration = time.perf_counter() - started
                self._results.put(AsyncResult(request, value, error, duration))
                self._busy.clear()

    def close(self, *, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if wait:
            while self.busy:
                self._thread.join(timeout=0.05)
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            # Only possible with wait=False; daemon shutdown will handle it.
            return
        if wait:
            self._thread.join(timeout=5.0)
