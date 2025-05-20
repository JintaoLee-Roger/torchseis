# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

import pynvml
import time
import threading


class GPUMemoryMonitor:
    """
    Usage:
    ```python
    monitor = GPUMemoryMonitor(gpu_id=1)
    monitor.start()
    # your code
    # xxxx
    gpu_peak = monitor.stop()
    print(f"GPU real peak: {gpu_peak / (1024**3):.2f} GB")
    ```
    """

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.peak_memory = 0
        self._stop_event = threading.Event()
        self._thread = None

    def _monitor(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        while not self._stop_event.is_set():
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.peak_memory = max(self.peak_memory, mem_info.used)
            time.sleep(0.01)  # 10ms采样间隔
        pynvml.nvmlShutdown()

    def start(self):
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        return self.peak_memory
