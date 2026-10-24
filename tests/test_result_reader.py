"""Only completed snapshots may appear in the results viewer."""
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from picstudio.post.reader import Results  # noqa: E402


class SnapshotReaderTests(unittest.TestCase):
    def test_snapshot_atomic_publish(self):
        for subdir, prefix, method in [("fields", "field", "field"),
                                       ("particles", "part", "particles")]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory) / subdir
                folder.mkdir()
                old = folder / f"{prefix}_0000001.npz"
                new = folder / f"{prefix}_0000002.npz"
                pending = folder / f"{new.name}.tmp.npz"
                np.savez(old, step=1)
                pending.write_bytes(b"unfinished compressed data")
                result = Results(directory)
                attr = "field_files" if method == "field" else "particle_files"
                self.assertEqual(getattr(result, attr), [str(old)])
                self.assertEqual(getattr(result, method)()["step"], 1)

                # Publish after the viewer enumerates: its existing selection stays valid.
                np.savez(pending, step=2)
                os.replace(pending, new)
                self.assertEqual(getattr(result, method)()["step"], 1)
                result.refresh()
                self.assertEqual(getattr(result, method)()["step"], 2)
                self.assertEqual(len(getattr(result, attr)), 2)

    def test_only_unfinished_snapshots(self):
        for subdir, prefix, method in [("fields", "field", "field"),
                                       ("particles", "part", "particles")]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory) / subdir
                folder.mkdir()
                (folder / f"{prefix}_0000001.npz.tmp.npz").write_bytes(b"unfinished")
                self.assertIsNone(getattr(Results(directory), method)())

    def test_corrupt_completed_snapshot_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "fields"
            folder.mkdir()
            (folder / "field_0000001.npz").write_bytes(b"corrupt")
            result = Results(directory)
            with self.assertRaises((ValueError, OSError, EOFError, pickle.UnpicklingError)):
                result.field()


if __name__ == "__main__":
    unittest.main()
