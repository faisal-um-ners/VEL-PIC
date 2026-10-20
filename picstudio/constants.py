"""Physical constants (CODATA 2018, SI units)."""
import math

c = 299792458.0                 # speed of light [m/s]
mu0 = 1.25663706212e-6          # vacuum permeability [H/m]
eps0 = 8.8541878128e-12         # vacuum permittivity [F/m]
e = 1.602176634e-19             # elementary charge [C]
me = 9.1093837015e-31           # electron mass [kg]
mp = 1.67262192369e-27          # proton mass [kg]
amu = 1.66053906660e-27         # atomic mass unit [kg]
kB = 1.380649e-23               # Boltzmann constant [J/K]
h = 6.62607015e-34              # Planck constant [J s]
hbar = h / (2 * math.pi)
eta0 = mu0 * c                  # free-space impedance [Ohm]
pi = math.pi

# Emission constants
A_RICHARDSON = 1.20173e6        # Richardson constant [A m^-2 K^-2]
FN_A = 1.541434e-6              # first Fowler-Nordheim constant [A eV V^-2]
FN_B = 6.830890e9               # second Fowler-Nordheim constant [eV^-3/2 V m^-1]
SCHOTTKY_C = 1.439964e-9        # e^3/(4 pi eps0) in [eV^2 m / V]

SPECIES_PRESETS = {
    # name: (charge in e, mass in kg)
    "electron": (-1.0, me),
    "positron": (1.0, me),
    "proton": (1.0, mp),
    "H+": (1.0, mp),
    "H2+": (1.0, 2.016 * amu),
    "He+": (1.0, 4.0026 * amu),
    "Ar+": (1.0, 39.948 * amu),
    "Xe+": (1.0, 131.293 * amu),
    "Cs+": (1.0, 132.905 * amu),
}

NAMESPACE = dict(c=c, mu0=mu0, eps0=eps0, e=e, qe=e, me=me, mp=mp, amu=amu, kB=kB,
                 h=h, hbar=hbar, eta0=eta0)
