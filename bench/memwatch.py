"""Peak process memory on macOS via proc_pid_rusage. phys_footprint includes Metal/GPU allocations (unified memory).

On Apple silicon the CPU and GPU share one pool of memory, so the process's physical footprint counts GPU tensors as well.
That makes it a fair way to compare memory use between PyTorch MPS and MLX, whose own memory counters measure different things.
"""
import ctypes, os, threading, time
_lib = ctypes.CDLL("/usr/lib/libproc.dylib")  # macOS system library that exposes per-process resource usage
_buf = (ctypes.c_uint64 * 64)()  # rusage_info_v4: 16-byte uuid, then uint64 fields

def _fields():
    """Fill _buf with this process's rusage_info_v4 (flavor 4) and return it as an array of uint64 words."""
    if _lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(_buf)) != 0:
        raise OSError("proc_pid_rusage failed")
    return _buf

def footprint():      # ri_phys_footprint: field 7 after the 2-word uuid
    """Current physical footprint in bytes."""
    return _fields()[2 + 7]

def lifetime_peak():  # ri_lifetime_max_phys_footprint: field 28
    """Highest physical footprint in bytes since the process started."""
    return _fields()[2 + 28]

class Watch:
    """Polls footprint on a thread; .peak is the max seen between start() and stop()."""
    def __init__(self, dt=0.02):  # dt: polling interval in seconds
        self.dt, self.peak, self._on = dt, 0, False
    def _run(self):
        while self._on:
            self.peak = max(self.peak, footprint()); time.sleep(self.dt)
    def start(self):
        # daemon=True: the polling thread never keeps the process alive on its own.
        self.peak, self._on = footprint(), True
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self
    def stop(self):
        self._on = False; self._t.join(); return self.peak
