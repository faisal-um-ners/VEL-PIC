"""python -m picstudio.conduction config.json --out results --frequency 1e9 ..."""
import argparse
import numpy as np
from . import ContactConfig, ContactFEM
from .plotting import save_results


def main():
    p=argparse.ArgumentParser(description="VEXPACE independent AC/DC conduction FEM")
    p.add_argument("config", help="JSON ContactConfig parameters, or {contact: {...}}")
    p.add_argument("--out",default="contact_results")
    p.add_argument("--frequency",type=float,action="append",default=None,
                   help="Individual frequency in Hz (repeatable); DC=0")
    p.add_argument("--fmin",type=float,default=1.0)
    p.add_argument("--fmax",type=float,default=1e12)
    p.add_argument("--points",type=int,default=31)
    a=p.parse_args()
    if a.frequency is not None:
        frequencies=np.array(a.frequency,dtype=float)
    else:
        frequencies=np.r_[0.0,np.geomspace(a.fmin,a.fmax,a.points)]
    c=ContactConfig.from_json(a.config)
    solver=ContactFEM(c)
    save_results(solver,frequencies,a.out)
    print("Conduction results:",a.out)
    for f in (0.0,frequencies[-1]):
        s=solver.solve(f)
        print(f"f={f:.6g} Hz, ZcW={s['Z_contact_W']*1e6:.6g} Ohm um, residual={s['relative_residual']:.3g}")


if __name__=="__main__":
    main()
