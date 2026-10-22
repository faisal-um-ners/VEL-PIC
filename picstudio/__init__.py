"""VEXPACE - Virtual Experiments in Plasma and Computational Electrodynamics (Python package ``picstudio``).

A particle-in-cell code for 1D (z), 2D Cartesian (z,y) and 2D axisymmetric (z,r) electrostatic and
electromagnetic simulations.

Copyright (c) 2026 Md Arifuzzaman Faisal. Released under the BSD 3-Clause License (see LICENSE).

Quick use from Python::

    from picstudio import load_model, Simulation
    sim = Simulation(load_model("examples/child_langmuir_1d.pic.json"))
    sim.run()
"""
__version__ = "1.3.0"
APP_NAME = "VEXPACE"
APP_TAGLINE = "Virtual Experiments in Plasma and Computational Electrodynamics"
APP_SUBTITLE = "An Open-Source Particle-in-Cell Simulation Framework for Plasma and Vacuum Electronics"
AUTHOR = "Md Arifuzzaman Faisal"
AFFILIATION = "Department of Nuclear Engineering and Radiological Sciences, University of Michigan, Ann Arbor, MI, USA"
COPYRIGHT = "Copyright (c) 2026 Md Arifuzzaman Faisal"
LICENSE = "BSD 3-Clause License"
URL = "https://github.com/faisal-um-ners/VEXPACE"
DOI = "10.5281/zenodo.23512471"          # Version-specific DOI for the archived Zenodo release
CITATION = ("Faisal, M. A. (2026). VEXPACE: An Open-Source Particle-in-Cell Simulation Framework for Plasma and "
            "Vacuum Electronics [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23512471")


def load_model(path):
    from .model.model import load
    return load(path)


def __getattr__(name):
    if name == "Simulation":
        from .core.simulation import Simulation
        return Simulation
    if name == "new_model":
        from .model.model import new_model
        return new_model
    raise AttributeError(name)
