"""Runs before anything imports numpy.

1. Starts numpy's math library (OpenBLAS) single-threaded. On some PCs it
   fails while setting up a thread per CPU core ("DLL load failed while
   importing _multiarray_umath"); the macro doesn't need more than one.
2. If files are missing or numpy still can't load, shows a clear message and
   writes startup_error.txt next to the exe instead of a raw crash.
"""
import glob
import os
import sys
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

APP = "Scroom Macro"


def _base():
    return os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))


def _fail(title, text, detail=""):
    path = os.path.join(_base(), "startup_error.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{text}\n\n--- details ---\n{detail}\n")
    except OSError:
        path = None
    msg = text + (f"\n\nDetails were saved to:\n{path}" if path else "")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(f"{APP} - {title}", msg)
        root.destroy()
    except Exception:
        pass
    sys.exit(1)


FIX_STEPS = (
    "How to fix it:\n"
    "1. Delete the ScroomMacro folder you extracted.\n"
    "2. Right-click the downloaded .zip -> Extract All... -> Extract.\n"
    "   (Don't run the app from inside the zip.)\n"
    "3. Check your antivirus' quarantine / Protection history - if it removed a\n"
    "   file from the ScroomMacro folder, restore it and allow the folder.\n"
    "4. Run App\\ScroomMacro.exe again."
)


def run():
    if getattr(sys, "frozen", False):
        internal = getattr(sys, "_MEIPASS", os.path.join(_base(), "_internal"))
        needed = {
            "numpy's math library (numpy.libs\\libopenblas...dll)": os.path.join(internal, "numpy.libs", "libopenblas*.dll"),
            "numpy core (numpy\\core\\_multiarray_umath...pyd)": os.path.join(internal, "numpy", "core", "_multiarray_umath*.pyd"),
        }
        missing = [name for name, pattern in needed.items() if not glob.glob(pattern)]
        if missing:
            _fail("files missing",
                  "Some files the app needs are missing:\n  - " + "\n  - ".join(missing) +
                  "\n\nThis usually means the zip wasn't fully extracted, or your antivirus "
                  "removed them.\n\n" + FIX_STEPS,
                  f"looked in: {internal}")
    try:
        import numpy  # noqa: F401
    except Exception:
        _fail("couldn't start",
              "Windows couldn't load numpy (a part the app uses to read the screen).\n\n"
              "This is usually caused by an incomplete extraction or by antivirus "
              "blocking one of its files.\n\n" + FIX_STEPS +
              "\n\nIf it still happens, send startup_error.txt to the macro's creator.",
              traceback.format_exc())
