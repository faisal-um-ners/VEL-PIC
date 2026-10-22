# Third-party notices

## WarpX

Selected particle-pushing, position-update, field-gathering and current-deposition
implementations in VEXPACE were translated or adapted from WarpX
(https://github.com/BLAST-WarpX/warpx). Other routines may implement the
same published mathematical methods without being direct source translations.
The relevant upstream sources and original contributor notices include:

- Boris momentum pusher: Source/Particles/Pusher/UpdateMomentumBoris.H
  Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe, Weiqun Zhang.
- Vay momentum pusher: Source/Particles/Pusher/UpdateMomentumVay.H
  Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe, Weiqun Zhang.
- Higuera-Cary pusher: Source/Particles/Pusher/UpdateMomentumHigueraCary.H
  Copyright 2019 Yinjian Zhao.
- Particle position update: Source/Particles/Pusher/UpdatePosition.H
  Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe, Weiqun Zhang.
- Field gathering: Source/Particles/Gather/FieldGather.H
  Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet,
  Revathi Jambunathan, Weiqun Zhang.
- Esirkepov current deposition: Source/Particles/Deposition/CurrentDeposition.H
  Copyright 2019 Axel Huebl, David Grote, Maxence Thevenet,
  Remi Lehe, Weiqun Zhang, Michael Rowan.

VEXPACE implementations are principally located in
`picstudio/particles/kernels.py` and `picstudio/fields/fdtd.py`.
The binomial current filter follows a method used in WarpX; whether its
source is directly translated has not been independently established.

These original contributor notices supplement the WarpX project license
reproduced below. WarpX's additional U.S. Government rights and
non-endorsement notice is preserved in `third_party/warpx/NOTICE.txt`.
VEXPACE is independently maintained and is not endorsed by WarpX's
developers or institutions.

WarpX is distributed under the following license:

```
WarpX Copyright (c) 2018, The Regents of the University of California,
through Lawrence Berkeley National Laboratory, and Lawrence Livermore National
Security, LLC, for the operation of Lawrence Livermore National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California,
Lawrence Berkeley National Laboratory, Lawrence Livermore National Security,
Lawrence Livermore National Laboratory, U.S. Dept. of Energy nor the names of
its contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source
code ("Enhancements") to anyone; however, if you choose to make your
Enhancements available either publicly, or directly to Lawrence Berkeley
National Laboratory, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license: a
non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works thereof,
in binary and source code form.
```

## Python dependencies

VEXPACE uses NumPy, SciPy, scikit-fem, Numba, Matplotlib and (optionally) PySide6 and mpi4py, each under
its own license. They are installed separately and are not redistributed with VEXPACE.
