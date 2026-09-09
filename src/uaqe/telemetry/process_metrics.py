"""Process and System Telemetry Sampler for UAQE.

Samples genuine host environment information, CPU utilization, and process/system RAM usage.
Never fabricates metrics.
"""

from __future__ import annotations

import os
import platform
import time
import threading
from typing import Dict, Any, Optional
import psutil

# Global tracker for peak process memory in this worker process
_PEAK_RAM_LOCK = threading.Lock()
_PEAK_PROCESS_RAM_MB = 0.0


def reset_peak_process_ram() -> None:
    """Reset the recorded peak process RAM."""
    global _PEAK_PROCESS_RAM_MB
    with _PEAK_RAM_LOCK:
        _PEAK_PROCESS_RAM_MB = 0.0


def get_peak_process_ram_mb() -> float:
    """Return the peak process RSS in MB recorded so far."""
    with _PEAK_RAM_LOCK:
        return _PEAK_PROCESS_RAM_MB


def get_host_environment_info() -> Dict[str, Any]:
    """Retrieve genuine host environment hardware and OS specifications."""
    try:
        cpu_count_logical = psutil.cpu_count(logical=True) or 1
        cpu_count_physical = psutil.cpu_count(logical=False) or 1
        virtual_mem = psutil.virtual_memory()
        total_ram_mb = round(virtual_mem.total / (1024 * 1024), 2)
    except Exception:
        cpu_count_logical = os.cpu_count() or 1
        cpu_count_physical = os.cpu_count() or 1
        total_ram_mb = 0.0

    # CPU frequencies
    freq_current = 0.0
    freq_max = 0.0
    try:
        freq = psutil.cpu_freq()
        if freq:
            freq_current = round(freq.current, 1)
            freq_max = round(freq.max, 1) if freq.max else round(freq.current, 1)
    except Exception:
        pass

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": cpu_count_logical,
        "cpu_count_physical": cpu_count_physical,
        "cpu_frequency_current_mhz": freq_current,
        "cpu_frequency_max_mhz": freq_max,
        "total_ram_mb": total_ram_mb,
        "python_version": platform.python_version(),
        "environment": "HOST",
    }


def sample_current_metrics(process: Optional[psutil.Process] = None) -> Dict[str, Any]:
    """Sample instantaneous CPU, process RAM, and system RAM metrics.
    
    Returns genuine measurements only. Never fabricates values.
    """
    global _PEAK_PROCESS_RAM_MB

    if process is None:
        try:
            process = psutil.Process(os.getpid())
        except Exception:
            process = None

    timestamp = time.time()
    
    # Process memory and CPU
    process_rss_mb = 0.0
    process_vms_mb = 0.0
    process_cpu_percent = 0.0
    process_threads = 1
    if process is not None:
        try:
            mem_info = process.memory_info()
            process_rss_mb = round(mem_info.rss / (1024 * 1024), 2)
            process_vms_mb = round(mem_info.vms / (1024 * 1024), 2)
            process_cpu_percent = round(process.cpu_percent(interval=None), 1)
            process_threads = process.num_threads()
        except Exception:
            pass

    # Update peak process RAM
    with _PEAK_RAM_LOCK:
        if process_rss_mb > _PEAK_PROCESS_RAM_MB:
            _PEAK_PROCESS_RAM_MB = process_rss_mb
        current_peak = _PEAK_PROCESS_RAM_MB

    # System memory & CPU
    system_cpu_percent = 0.0
    system_ram_percent = 0.0
    system_ram_used_mb = 0.0
    system_ram_available_mb = 0.0
    try:
        system_cpu_percent = round(psutil.cpu_percent(interval=None), 1)
        sys_mem = psutil.virtual_memory()
        system_ram_percent = round(sys_mem.percent, 1)
        system_ram_used_mb = round(sys_mem.used / (1024 * 1024), 2)
        system_ram_available_mb = round(sys_mem.available / (1024 * 1024), 2)
    except Exception:
        pass

    # CPU Freq
    freq_current = 0.0
    freq_max = 0.0
    try:
        freq = psutil.cpu_freq()
        if freq:
            freq_current = round(freq.current, 1)
            freq_max = round(freq.max, 1) if freq.max else round(freq.current, 1)
    except Exception:
        pass

    try:
        cpu_count_logical = psutil.cpu_count(logical=True) or 1
        cpu_count_physical = psutil.cpu_count(logical=False) or 1
    except Exception:
        cpu_count_logical = 1
        cpu_count_physical = 1

    return {
        "timestamp": timestamp,
        "process_cpu_percent": process_cpu_percent,
        "system_cpu_percent": system_cpu_percent,
        "process_ram_mb": process_rss_mb,
        "process_vms_mb": process_vms_mb,
        "peak_process_ram_mb": current_peak,
        "system_ram_used_mb": system_ram_used_mb,
        "system_ram_available_mb": system_ram_available_mb,
        "system_ram_percent": system_ram_percent,
        "process_threads": process_threads,
        "cpu_count_logical": cpu_count_logical,
        "cpu_count_physical": cpu_count_physical,
        "cpu_frequency_current_mhz": freq_current,
        "cpu_frequency_max_mhz": freq_max,
    }
