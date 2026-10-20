"""Loss-port loaded Q of the magnetron examples (cold-test ring-down), and the A6 pi-mode frequency.

    python benchmarks/bench_qport.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, record, row  # noqa: E402
from picstudio.core.qcal import measure_q  # noqa: E402
from picstudio.model.model import evaluator, load  # noqa: E402


def main():
    rows = []
    for fn, label in (("planar_magnetron_em.pic.json", "Planar magnetron"),
                      ("a6_magnetron_em.pic.json", "A6 magnetron")):
        m = load(os.path.join(ROOT, "examples", fn))
        ev = evaluator(m)
        obj = [o for o in m["objects"] if o.get("material") == "lossy"][0]
        Qt, f0 = ev.num(obj["Q"]), ev.num(obj["frequency"])
        Q, f = measure_q(m, f0)
        rows.append(row(f"{label}: loaded Q of the loss port (target {Qt:g})", Q, Qt, 0.05))
        if label.startswith("A6"):
            rows.append(row("A6 cold-test pi-mode frequency (GHz) vs ~2.34 GHz (literature)", f / 1e9, 2.34, 0.03))
    record("23_loss_port_q", "Loss port with defined loaded Q (cold-test ring-down)", rows,
           notes="The lossy load has sigma = 2 pi f eps0/(Q F); F (electric-energy fill factor) was calibrated "
                 "with 'picstudio qcal' and the loaded Q re-measured here by an independent ring-down.")


if __name__ == "__main__":
    main()
