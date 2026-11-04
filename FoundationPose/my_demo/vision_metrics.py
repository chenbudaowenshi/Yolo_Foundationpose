"""Session telemetry for measured vision operations and actual publication recovery."""
from __future__ import annotations

from collections import Counter, deque
import json
import math
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np


class Distribution:
    def __init__(self, capacity=100000):
        self.values = deque(maxlen=capacity)
        self.count = 0
        self.maximum = None

    def add(self, value):
        if value is None or not math.isfinite(value):
            return
        value = float(value)
        self.count += 1
        self.values.append(value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def snapshot(self):
        quantiles = np.percentile(self.values, [50, 95]).tolist() if self.values else [None, None]
        return {"count": self.count, "p50": quantiles[0], "p95": quantiles[1], "max": self.maximum,
                "quantile_samples": len(self.values),
                "quantile_scope": "whole_run" if self.count == len(self.values) else "latest_bounded_samples"}


class ProcessGpuSampler:
    """Sample this PID through NVML, including allocations outside PyTorch.

    This is an observed/sampled peak, not a claim that sub-interval spikes are
    captured. Other processes' GPU usage is never added to this measurement.
    """

    def __init__(self, interval_s=.05):
        self.interval_s = interval_s
        self.pid = os.getpid()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.data = {"source": "NVML process memory", "pid": self.pid,
                     "sampling_interval_s": interval_s, "available": False,
                     "sample_count": 0, "process_peak_bytes": None, "devices": [], "error": None}
        self.thread = threading.Thread(target=self._run, name="vision-gpu-memory", daemon=True)
        self.thread.start()

    def _run(self):
        nv = None
        try:
            import pynvml as nv
            nv.nvmlInit()
            devices = []
            for index in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(index)
                def decode(value): return value.decode() if isinstance(value, bytes) else str(value)
                devices.append((handle, {"index": index, "name": decode(nv.nvmlDeviceGetName(handle)),
                                         "uuid": decode(nv.nvmlDeviceGetUUID(handle)), "process_peak_bytes": 0}))
            while True:
                total = 0
                for handle, device in devices:
                    values = []
                    successful_queries = 0
                    for query in (nv.nvmlDeviceGetComputeRunningProcesses, nv.nvmlDeviceGetGraphicsRunningProcesses):
                        try:
                            processes = query(handle)
                            successful_queries += 1
                        except nv.NVMLError:
                            continue
                        for process in processes:
                            if process.pid == self.pid:
                                value = process.usedGpuMemory
                                if value is None or value >= (1 << 63):
                                    raise RuntimeError("NVML cannot report per-process memory on this device")
                                values.append(int(value))
                    if not successful_queries:
                        raise RuntimeError("NVML process queries unavailable")
                    # The same PID can be in both lists; count it once per GPU.
                    used = max(values, default=0)
                    total += used
                    device["process_peak_bytes"] = max(device["process_peak_bytes"], used)
                with self.lock:
                    self.data.update(available=True, sample_count=self.data["sample_count"]+1,
                                     process_peak_bytes=max(self.data["process_peak_bytes"] or 0, total),
                                     devices=[dict(d) for _, d in devices])
                if self.stop_event.wait(self.interval_s):
                    break
        except Exception as exc:
            with self.lock:
                self.data["error"] = str(exc)
        finally:
            if nv is not None:
                try: nv.nvmlShutdown()
                except Exception: pass

    def snapshot(self):
        with self.lock:
            return dict(self.data, devices=[dict(d) for d in self.data["devices"]])

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


class VisionMetrics:
    def __init__(self, context, *, now_ns=None):
        self.session_id = str(uuid4())
        self.context = dict(context)
        self.started_ns = time.monotonic_ns() if now_ns is None else now_ns
        self.operations = {}
        self.frames = Counter()
        self.recovery_duration = Distribution()
        self.model_load_duration = Distribution()
        self.published_delay = Distribution()
        self.recovery_by_reason = {}
        self.recoveries = []
        self.recovery_count = 0
        self.active_recovery = None
        self.first_publication_s = None
        self.last_publication_ns = None
        self.torch_memory = None
        self.begin_recovery("startup", self.started_ns)

    def begin_recovery(self, reason, now_ns=None, capture_stamp_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if self.active_recovery is None:
            self.recovery_count += 1
            self.active_recovery = {"recovery_id": self.recovery_count, "reason": reason,
                "started_mono_ns": now_ns, "trigger_capture_stamp_ns": capture_stamp_ns, "triggers": [reason]}
        elif reason not in self.active_recovery["triggers"]:
            # Repeated failures must not move the start forward and hide downtime.
            self.active_recovery["triggers"].append(reason)
        return self.active_recovery["recovery_id"]

    def record(self, record, *, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        record.update(schema_version=2, record_type="frame", session_id=self.session_id)
        mode = record.get("mode", "none")
        if record.get("gpu_memory"):
            previous = self.torch_memory or {}
            self.torch_memory = dict(previous, **record["gpu_memory"])
            if record["gpu_memory"].get("available") is True:
                self.torch_memory.pop("error", None)
            for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                if key in previous:
                    self.torch_memory[key] = max(previous[key], self.torch_memory.get(key, 0))
        self.frames[record["status"]] += 1
        self.model_load_duration.add(record.get("timings", {}).get("model_load_s"))
        if record.get("published"):
            self.published_delay.add(record.get("observation_delay_s"))
        group = self.operations.setdefault(mode, {"attempts": 0, "valid_estimates": 0,
                    "published": 0, "full_search_requests": 0, "full_search_executions": 0,
                    "reasons": Counter(), "durations": {}})
        group["attempts"] += 1
        group["valid_estimates"] += record.get("estimate_valid", False)
        group["published"] += record.get("published", False)
        group["full_search_requests"] += record.get("full_search_requested") is True
        group["full_search_executions"] += record.get("search", {}).get("full_search") is True
        group["reasons"][record.get("trigger_reason") or "none"] += 1
        for name, value in {**record.get("timings", {}), "worker_s": record.get("worker_s"),
                            "observation_delay_s": record.get("observation_delay_s")}.items():
            group["durations"].setdefault(name, Distribution()).add(value)
        if record.get("status") == "shutdown_discard":
            return record
        trigger = record.get("trigger_mono_ns")
        if record.get("operation") == "register" and trigger is not None:
            self.begin_recovery(record.get("trigger_reason") or "registration", trigger, record.get("stamp_ns"))
            start = record.get("operation_start_mono_ns")
            if start is not None:
                self.active_recovery.setdefault("first_registration_start_mono_ns", start)
                self.active_recovery["last_registration_start_mono_ns"] = start
        if not record.get("published"):
            self.begin_recovery(record.get("status", "unpublished"),
                                record.get("processing_start_mono_ns") or now_ns, record.get("stamp_ns"))
        else:
            self.last_publication_ns = now_ns
            if self.first_publication_s is None:
                self.first_publication_s = (now_ns-self.started_ns)/1e9
            if self.active_recovery is not None:
                recovery = dict(self.active_recovery, completed_mono_ns=now_ns,
                                first_valid_sequence=record["sequence"], first_valid_stamp_ns=record["stamp_ns"],
                                duration_s=(now_ns-self.active_recovery["started_mono_ns"])/1e9)
                for key in ("first_registration_start_mono_ns", "last_registration_start_mono_ns"):
                    if key in recovery:
                        recovery[key.replace("_mono_ns", "_to_output_s")] = (now_ns-recovery[key])/1e9
                self.recovery_duration.add(recovery["duration_s"])
                self.recovery_by_reason.setdefault(recovery["reason"], Distribution()).add(recovery["duration_s"])
                self.recoveries.append(recovery)
                self.recoveries = self.recoveries[-1024:]
                record["completed_recovery"] = recovery
                self.active_recovery = None
        record["active_recovery_id"] = self.active_recovery["recovery_id"] if self.active_recovery else None
        return record

    def check_output_timeout(self, maximum_gap_s, *, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        if self.last_publication_ns is None or self.active_recovery is not None:
            return False
        deadline = self.last_publication_ns + int(maximum_gap_s*1e9)
        if now_ns > deadline:
            self.begin_recovery("output_timeout", deadline)
            return True
        return False

    def snapshot(self, *, gpu=None, final=False, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        operations = {}
        for mode, group in self.operations.items():
            operations[mode] = {**group, "reasons": dict(group["reasons"]),
                               "durations_seconds": {k:v.snapshot() for k,v in group["durations"].items()}}
            operations[mode].pop("durations")
        for mode in ("initial_registration", "reregistration", "tracking"):
            operations.setdefault(mode, {"attempts": 0, "valid_estimates": 0, "published": 0,
                "full_search_requests": 0, "full_search_executions": 0, "reasons": {},
                "durations_seconds": {"pose_s": Distribution().snapshot()}})
        pending = None if self.active_recovery is None else dict(self.active_recovery,
                         elapsed_s=(now_ns-self.active_recovery["started_mono_ns"])/1e9)
        return {"schema_version": 2, "session_id": self.session_id, "context": self.context,
                "final": final, "runtime_s": (now_ns-self.started_ns)/1e9,
                "time_to_first_valid_output_s": self.first_publication_s,
                "model_load_seconds": self.model_load_duration.snapshot(),
                "published_observation_delay_seconds": self.published_delay.snapshot(),
                "operations": operations, "frame_status_counts": dict(self.frames),
                "recovery_duration_seconds": self.recovery_duration.snapshot(),
                "recovery_by_trigger_seconds": {k:v.snapshot() for k,v in self.recovery_by_reason.items()},
                "pending_recovery": pending, "recent_completed_recoveries": self.recoveries,
                "gpu_memory": {"nvml_sampled": gpu, "torch_allocator": self.torch_memory}}


def write_summary(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)
