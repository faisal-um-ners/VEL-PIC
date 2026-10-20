"""Minimal reader for AMReX plotfiles (single level) used to compare with WarpX."""
import os
import re

import numpy as np


def read_plotfile(path):
    with open(os.path.join(path, "Header")) as fh:
        lines = [l.rstrip("\n") for l in fh]
    k = 1
    nvar = int(lines[k]); k += 1
    names = lines[k:k + nvar]; k += nvar
    dim = int(lines[k]); k += 1
    time = float(lines[k]); k += 1
    k += 1                                   # finest level
    lo = [float(x) for x in lines[k].split()]; k += 1
    hi = [float(x) for x in lines[k].split()]; k += 1
    k += 1                                   # ref ratio (empty)
    dom = lines[k]; k += 1
    nums = [int(x) for x in re.findall(r"-?\d+", dom)]
    ncell = [nums[dim + i] - nums[i] + 1 for i in range(dim)]
    # level 0 cell header
    with open(os.path.join(path, "Level_0", "Cell_H")) as fh:
        ch = fh.read().splitlines()
    ncomp = int(ch[2])
    fabs = [l.split() for l in ch if l.startswith("FabOnDisk")]
    nbox = int(re.findall(r"\((\d+)", ch[4])[0])
    boxes = []
    for i in range(nbox):
        b = [int(x) for x in re.findall(r"-?\d+", ch[5 + i])]
        boxes.append((b[:dim], b[dim:2 * dim]))
    data = np.zeros([ncomp] + ncell)
    for (blo, bhi), f in zip(boxes, fabs):
        fname, off = f[1], int(f[2])
        with open(os.path.join(path, "Level_0", fname), "rb") as fh:
            fh.seek(off)
            hdr = fh.readline().decode()
            order = re.findall(r"\(8, \(([\d ]+)\)\)", hdr)[1].split()
            dt = "<f8" if order[0] == "8" else ">f8"
            shape = [bhi[i] - blo[i] + 1 for i in range(dim)]
            npts = int(np.prod(shape))
            arr = np.frombuffer(fh.read(8 * npts * ncomp), dtype=dt).reshape([ncomp] + shape[::-1])
            arr = np.transpose(arr, [0] + list(range(dim, 0, -1)))
            sl = tuple(slice(blo[i], bhi[i] + 1) for i in range(dim))
            data[(slice(None),) + sl] = arr
    out = {n: data[i] for i, n in enumerate(names)}
    coords = [lo[i] + (np.arange(ncell[i]) + 0.5) * (hi[i] - lo[i]) / ncell[i] for i in range(dim)]
    return out, coords, time
