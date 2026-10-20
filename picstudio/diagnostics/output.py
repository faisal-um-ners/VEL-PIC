"""Output writer: history (CSV), field/particle snapshots (NPZ), live file for the GUI."""
from __future__ import annotations

import json
import os
import time

import numpy as np


class Output:
    def __init__(self, out_dir: str, enabled: bool = True):
        self.dir = out_dir
        self.enabled = enabled
        self._hist = None
        self.columns = []
        if enabled:
            os.makedirs(os.path.join(out_dir, "fields"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "particles"), exist_ok=True)
            for sub in ("fields", "particles"):
                for f in os.listdir(os.path.join(out_dir, sub)):
                    if f.endswith(".npz"):
                        _remove_retry(os.path.join(out_dir, sub, f))
            for flag in ("STOP", "PAUSE"):
                fp = os.path.join(out_dir, flag)
                if os.path.exists(fp):
                    _remove_retry(fp)

    def start_history(self, columns):
        self.columns = list(columns)
        if not self.enabled:
            return
        self._hist = open(os.path.join(self.dir, "history.csv"), "w", encoding="utf-8")
        self._hist.write(",".join(self.columns) + "\n")
        self._hist.flush()

    def history(self, row):
        if self._hist is None:
            return
        self._hist.write(",".join(f"{v:.9g}" for v in row) + "\n")
        self._hist.flush()

    def save_npz(self, sub, name, data, optional=False):
        """Write an NPZ file atomically (temporary file + rename).

        On Windows a file cannot be replaced while another process (the GUI reading the live file,
        an antivirus scanner, the search indexer, a cloud-sync client) has it open, and os.replace
        raises PermissionError.  The rename is retried for about 2 s; an optional file (live.npz,
        which only feeds the live monitor) is skipped for this update if it is still locked, so the
        run never stops because of it."""
        if not self.enabled:
            return
        path = os.path.join(self.dir, sub, name) if sub else os.path.join(self.dir, name)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **data)
        if _replace_retry(tmp, path):
            return
        if optional:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return
        os.replace(tmp, path)              # last attempt: raise the real error

    def write_json(self, name, data):
        if not self.enabled:
            return
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_jsonable)

    def stop_requested(self):
        return self.enabled and os.path.exists(os.path.join(self.dir, "STOP"))

    def pause_requested(self):
        return self.enabled and os.path.exists(os.path.join(self.dir, "PAUSE"))

    def close(self):
        if self._hist is not None:
            self._hist.close()
            self._hist = None


def _replace_retry(tmp, path, tries=40, wait=0.05):
    """os.replace with retries for files briefly locked by another process (Windows)."""
    for _ in range(tries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(wait)
    return False


def _remove_retry(path, tries=40, wait=0.05):
    """os.remove with retries for files briefly locked by another process (Windows).  A file that
    stays locked is left in place (an old snapshot is overwritten later or ignored)."""
    for _ in range(tries):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            time.sleep(wait)
    return False


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
