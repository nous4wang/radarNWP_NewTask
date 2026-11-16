"""E42 follow-up -- call the NVML library directly to read the error code.

READ-ONLY.  It loads libnvidia-ml.so.1 through ctypes and calls
``nvmlInit()``, whose return code says what the monitoring layer objected
to.  ``nvidia-smi`` only prints "Unknown Error" and hides the code, so the
library is asked directly.

Nothing is trained, allocated on a GPU or written.
"""
from __future__ import annotations

import ctypes
import sys

# NVML error codes that matter here.
NVML_SUCCESS = 0
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_UNKNOWN = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_LIBRARY_NOT_FOUND = 12


def main() -> int:
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError as exc:
        print(f"libnvidia-ml.so.1 could not be loaded: {exc}")
        print("VERDICT: NVML LIBRARY MISSING (driver not installed for this user)")
        return 1

    lib.nvmlInit.restype = ctypes.c_int
    rc = lib.nvmlInit()
    names = {
        NVML_SUCCESS: "NVML_SUCCESS",
        NVML_ERROR_UNKNOWN: "NVML_ERROR_UNKNOWN",
        NVML_ERROR_NO_PERMISSION: "NVML_ERROR_NO_PERMISSION",
        NVML_ERROR_DRIVER_NOT_LOADED: "NVML_ERROR_DRIVER_NOT_LOADED",
        NVML_ERROR_LIBRARY_NOT_FOUND: "NVML_ERROR_LIBRARY_NOT_FOUND",
    }
    print(f"nvmlInit() rc = {rc} ({names.get(rc, 'unmapped')})")

    if rc == NVML_SUCCESS:
        count = ctypes.c_uint()
        lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        print(f"device count = {count.value}")
        for i in range(count.value):
            handle = ctypes.c_void_p()
            lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle))
            name = ctypes.create_string_buffer(96)
            lib.nvmlDeviceGetName(handle, name, 96)
            mem = ctypes.c_ulonglong()
            lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
            print(f"  cuda:{i} {name.value.decode()} free_mem={mem.value // (1<<20)} MiB")
        lib.nvmlShutdown()
        print("VERDICT: NVML OK")
        return 0

    if rc == NVML_ERROR_UNKNOWN:
        print("NVML_ERROR_UNKNOWN means the driver is loaded but a device is in "
              "a bad state (a GPU fell off the bus).  This needs a host-level "
              "reset; it cannot be fixed from inside the container.")
    print("VERDICT: NVML INIT FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
