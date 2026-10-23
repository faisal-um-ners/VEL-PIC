"""Independent 2D anisotropic AC/DC contact-conduction FEM.

This is NOT a PIC/Poisson/Yee solver. It uses separate, duplicated P1 nodes at
the contact interface and a parallel-RC interface admittance. Coordinates use
(y,z), matching Figure_3.ipynb and Figure_S1_S2.ipynb reference notebooks.

Convention: phasors exp(+i omega t); J = -(sigma+i omega epsilon) grad(phi).
All contact impedances are multiplied by device width (Ohm*m).
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

EPS0 = 8.8541878128e-12


@dataclass
class ContactConfig:
    # Example geometry ONLY: overwrite these with measured material properties.
    contact_length: float = 500e-9
    exposed_length: float = 1000e-9    # full channel length; half is simulated
    metal_thickness: float = 50e-9
    semiconductor_thickness: float = 10e-9
    rho1_y: float = 2.5e-8             # bulk resistivities: ohm*m
    rho1_z: float = 2.5e-8
    rho2_y: float = 2e-4
    rho2_z: float = 2e-3
    rho_i: float = 1e-12               # specific interface ohm*m^2
    c_i: float = 0.0                   # specific parallel capacitance F/m^2
    eps1_y: float = 1.0                # relative permittivities
    eps1_z: float = 1.0
    eps2_y: float = 6.0
    eps2_z: float = 6.0
    v_metal: float = 1.0               # normalized terminal potentials
    v_channel: float = 0.0
    n_contact: int = 70
    n_exposed: int = 50
    n_metal_depth: int = 20
    n_sem_depth: int = 20

    def validate(self):
        positives = ("contact_length", "metal_thickness", "semiconductor_thickness",
                     "rho1_y", "rho1_z", "rho2_y", "rho2_z", "rho_i")
        for key in positives:
            if getattr(self, key) <= 0:
                raise ValueError(key + " must be positive")
        for key in ("c_i", "exposed_length"):
            if getattr(self, key) < 0:
                raise ValueError(key + " cannot be negative")
        for key in ("n_contact", "n_metal_depth", "n_sem_depth"):
            if getattr(self, key) < 2:
                raise ValueError(key + " must be >= 2")
        if self.exposed_length > 0 and self.n_exposed < 2:
            raise ValueError("n_exposed must be >= 2")
        if self.v_metal == self.v_channel:
            raise ValueError("terminals must have different potentials")

    @classmethod
    def from_json(cls, path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**obj.get("contact", obj))


def _region(ys, zs, offset):
    y, z = np.meshgrid(ys, zs, indexing="ij")
    xyz = np.column_stack((y.ravel(), z.ravel()))
    ids = np.arange(offset, offset + len(xyz)).reshape(len(ys), len(zs))
    tris = []
    for i in range(len(ys) - 1):
        for j in range(len(zs) - 1):
            a, b, c, d = ids[i, j], ids[i+1, j], ids[i, j+1], ids[i+1, j+1]
            if (i+j) % 2:
                tris.extend(((a,b,c), (b,d,c)))
            else:
                tris.extend(((a,b,d), (a,d,c)))
    return xyz, np.asarray(tris, dtype=np.int64), ids


class ContactFEM:
    """Cache mesh and P1 triangle gradients; change only admittivities per f."""

    def __init__(self, config):
        self.cfg = c = config if isinstance(config, ContactConfig) else ContactConfig(**config)
        c.validate()
        t = np.linspace(0.0, 1.0, c.n_contact + 1)
        yc = c.contact_length * (1.0 - (1.0 - t)**2)
        if c.exposed_length:
            t = np.linspace(0.0, 1.0, c.n_exposed + 1)[1:]
            ys = np.r_[yc, c.contact_length + 0.5*c.exposed_length*t*t]
        else:
            ys = yc.copy()
        zm = c.metal_thickness * np.linspace(0.0, 1.0, c.n_metal_depth+1)**1.8
        t = np.linspace(0.0, 1.0, c.n_sem_depth+1)
        zs = -c.semiconductor_thickness + c.semiconductor_thickness*(1.0-(1.0-t)**1.8)
        pm, tm, im = _region(yc, zm, 0)
        ps, ts, is_ = _region(ys, zs, len(pm))
        self.points = np.vstack((pm, ps))
        self.triangles = np.vstack((tm, ts))
        self.material = np.r_[np.ones(len(tm), np.int8), np.full(len(ts), 2, np.int8)]
        self.ids_metal, self.ids_sem = im, is_
        self.y_contact, self.y_sem, self.z_metal, self.z_sem = yc, ys, zm, zs
        self.i_top, self.i_bottom = im[:, 0], is_[:len(yc), -1]
        self.i_metal_contact = im[:, -1]
        self.i_channel_terminal = is_[-1, :]
        self.n = len(self.points)
        self._geometry()

    def _geometry(self):
        p = self.points[self.triangles]
        det = ((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1]) -
               (p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1]))
        if np.any(abs(det) < 1e-60):
            raise ValueError("degenerate FEM element")
        self.area = abs(det)/2
        self.grad = np.stack((
            np.stack((p[:,1,1]-p[:,2,1], p[:,2,0]-p[:,1,0]),axis=1),
            np.stack((p[:,2,1]-p[:,0,1], p[:,0,0]-p[:,2,0]),axis=1),
            np.stack((p[:,0,1]-p[:,1,1], p[:,1,0]-p[:,0,0]),axis=1)
        ),axis=1)/det[:,None,None]
        self.rows = np.repeat(self.triangles, 3, axis=1).ravel()
        self.cols = np.tile(self.triangles, (1, 3)).ravel()
        self.fixed = np.unique(np.r_[self.i_metal_contact, self.i_channel_terminal])
        self.free = np.setdiff1d(np.arange(self.n), self.fixed)
        self.bval = np.zeros(self.n, dtype=complex)
        c = self.cfg
        self.bval[self.i_metal_contact] = c.v_metal
        self.bval[self.i_channel_terminal] = c.v_channel

    def solve(self, frequency_hz=0.0):
        c = self.cfg
        f = float(frequency_hz)
        if f < 0:
            raise ValueError("frequency must be >= 0")
        w = 2*np.pi*f
        eta = (
            (1/c.rho1_y + 1j*w*EPS0*c.eps1_y, 1/c.rho1_z + 1j*w*EPS0*c.eps1_z),
            (1/c.rho2_y + 1j*w*EPS0*c.eps2_y, 1/c.rho2_z + 1j*w*EPS0*c.eps2_z),
        )
        ey = np.where(self.material == 1, eta[0][0], eta[1][0])
        ez = np.where(self.material == 1, eta[0][1], eta[1][1])
        g = self.grad
        local = self.area[:,None,None] * (
            ey[:,None,None]*g[:,:,0,None]*g[:,None,:,0] +
            ez[:,None,None]*g[:,:,1,None]*g[:,None,:,1])
        k = coo_matrix((local.ravel(), (self.rows, self.cols)), shape=(self.n,self.n)).tocsr()
        yint = 1/c.rho_i + 1j*w*c.c_i
        base = np.array(((2,1,-2,-1),(1,2,-1,-2),(-2,-1,2,1),(-1,-2,1,2)))
        ids = np.column_stack((self.i_top[:-1],self.i_top[1:],
                               self.i_bottom[:-1],self.i_bottom[1:]))
        seg = np.diff(self.y_contact)
        interface = (seg[:,None,None]*yint/6)*base[None,:,:]
        k = k + coo_matrix((interface.ravel(), (np.repeat(ids,4,axis=1).ravel(),
                               np.tile(ids,(1,4)).ravel())),shape=(self.n,self.n)).tocsr()
        v = self.bval.copy()
        v[self.free] = spsolve(k[self.free][:,self.free],
                              -(k[self.free][:,self.fixed]@v[self.fixed]))
        jump = v[self.i_top]-v[self.i_bottom]
        # Linear P1 trace integration is exact with the trapezoidal rule.
        current_per_width = yint*np.trapezoid(jump,self.y_contact)
        if abs(current_per_width) < 1e-300:
            raise ZeroDivisionError("zero interface current")
        total_zw = (c.v_metal-c.v_channel)/current_per_width
        access_zw = (0.5*c.exposed_length)/(c.semiconductor_thickness*eta[1][0])
        contact_zw = total_zw-access_zw
        gradv = np.einsum("ti,tij->tj",v[self.triangles],self.grad)
        j = -np.column_stack((ey*gradv[:,0],ez*gradv[:,1]))
        e = -gradv
        # 1/2 Re(eta)|E|^2 dissipated per unit device width.
        power_bulk = 0.5*np.sum(self.area * (np.real(ey)*abs(e[:,0])**2 +
                                             np.real(ez)*abs(e[:,1])**2))
        # Exact integral of |linear jump|^2 on interface segments.
        a,b=jump[:-1],jump[1:]
        integ=np.sum(seg*(abs(a)**2+np.real(a*np.conj(b))+abs(b)**2)/3)
        power_interface=0.5*np.real(yint)*integ
        residual = np.linalg.norm((k@v)[self.free]) / max(np.linalg.norm(k[self.free][:,self.fixed]@v[self.fixed]),1e-40)
        return {"frequency_hz":f,"omega":w, "Z_total_W":total_zw,
                "Z_contact_W":contact_zw,"Z_access_W":access_zw,
                "interface_admittance":yint,
                "interface_specific_impedance":1/yint,
                "current_per_width":current_per_width,"potential":v,
                "J_elements":j,"E_elements":e,"jump":jump,
                "J_interface":yint*jump,"power_bulk_per_width":power_bulk,
                "power_interface_per_width":power_interface,
                "relative_residual":residual}

    def sweep(self, frequencies):
        rows=[]
        for f in frequencies:
            s=self.solve(f)
            z=s["Z_contact_W"]
            rows.append((f,z.real,z.imag,abs(z),np.angle(z,deg=True),
                         s["Z_total_W"].real,s["Z_total_W"].imag,
                         s["interface_specific_impedance"].real,
                         s["interface_specific_impedance"].imag))
        return np.asarray(rows,float)

SWEEP_COLUMNS=("frequency_hz","Re_ZcontactW_ohm_m","Im_ZcontactW_ohm_m",
               "abs_ZcontactW_ohm_m","phase_Zcontact_deg",
               "Re_ZtotalW_ohm_m","Im_ZtotalW_ohm_m",
               "Re_zi_ohm_m2","Im_zi_ohm_m2")
