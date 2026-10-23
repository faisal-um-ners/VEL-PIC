"""Plots and machine-readable exports for anisotropic contact conduction."""
from pathlib import Path
import json
import numpy as np
from .solver import SWEEP_COLUMNS


def save_results(solver, frequencies, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = Path(destination)
    d.mkdir(parents=True, exist_ok=True)
    (d / "parameters.json").write_text(
        json.dumps(vars(solver.cfg), indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.triplot(solver.points[:, 0] * 1e9, solver.points[:, 1] * 1e9,
               solver.triangles, lw=0.25, color="gray")
    ax.set(xlabel="Contact direction y (nm)", ylabel="Vertical z (nm)", title="Interface-refined FEM mesh")
    fig.tight_layout()
    fig.savefig(d / "mesh.png", dpi=240)
    plt.close(fig)

    sweep = solver.sweep(frequencies)
    np.savetxt(d / "impedance_sweep.csv", sweep, delimiter=",",
               header=",".join(SWEEP_COLUMNS), comments="")
    if len(sweep) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        positive = sweep[:, 0] > 0
        if np.any(positive):
            ax.semilogx(sweep[positive,0], 1e6*sweep[positive,1], label="Re ZcW")
            ax.semilogx(sweep[positive,0], 1e6*sweep[positive,2], label="Im ZcW")
            ax.semilogx(sweep[positive,0], 1e6*sweep[positive,3], label="|ZcW|")
        ax.set(xlabel="Frequency (Hz)", ylabel="Contact impedance × width (Ohm µm)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(d / "impedance_vs_frequency.png", dpi=240)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        if np.any(positive):
            ax.semilogx(sweep[positive,0], sweep[positive,4])
        ax.set(xlabel="Frequency (Hz)", ylabel="Phase of Zc (degrees)")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(d / "phase_vs_frequency.png", dpi=240)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(1e6*sweep[:,1], -1e6*sweep[:,2], "o-")
        ax.set(xlabel="Re ZcW (Ohm µm)", ylabel="-Im ZcW (Ohm µm)", title="Contact impedance (Nyquist)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(d / "nyquist.png", dpi=240)
        plt.close(fig)

    for f in sorted(set([0.0, float(frequencies[-1])])):
        solution=solver.solve(f)
        prefix=("dc" if f==0 else "ac_"+format(f,".4g").replace("+",""))
        for quantity in ("potential","J_elements","E_elements","jump","J_interface"):
            np.save(d / (prefix+"_"+quantity+".npy"), solution[quantity])
        info={key: [float(np.real(v)),float(np.imag(v))] for key,v in solution.items()
              if key in ("Z_total_W","Z_contact_W","Z_access_W","current_per_width",
                         "interface_specific_impedance")}
        info.update({"frequency_hz":f,"residual":float(solution["relative_residual"]),
                     "power_bulk_per_width":float(solution["power_bulk_per_width"]),
                     "power_interface_per_width":float(solution["power_interface_per_width"])})
        (d / (prefix+"_summary.json")).write_text(json.dumps(info,indent=2),encoding="utf-8")
        for part in ("real", "magnitude"):
            vals=(np.real(solution["potential"]) if part=="real" else abs(solution["potential"]))
            fig,ax=plt.subplots(figsize=(8,3.5))
            m=ax.tripcolor(solver.points[:,0]*1e9,solver.points[:,1]*1e9,
                           solver.triangles, vals, shading="gouraud")
            fig.colorbar(m,ax=ax,label="Potential (V)")
            ax.set(xlabel="y (nm)",ylabel="z (nm)",title=f"{prefix}: {part} potential")
            fig.tight_layout()
            fig.savefig(d/(prefix+"_potential_"+part+".png"),dpi=240)
            plt.close(fig)
        for part in ("magnitude","real"):
            j=solution["J_elements"]
            value=(np.linalg.norm(j,axis=1) if part=="magnitude"
                   else np.linalg.norm(np.real(j),axis=1))
            fig,ax=plt.subplots(figsize=(8,3.5))
            m=ax.tripcolor(solver.points[:,0]*1e9,solver.points[:,1]*1e9,
                           solver.triangles,facecolors=value,shading="flat")
            fig.colorbar(m,ax=ax,label="|J| (A/m²)")
            ax.set(xlabel="y (nm)",ylabel="z (nm)",title=f"{prefix}: {part} current density")
            fig.tight_layout()
            fig.savefig(d/(prefix+"_current_"+part+".png"),dpi=240)
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(solver.y_contact*1e9,np.abs(solution["J_interface"]),label="|J interface|")
        ax.plot(solver.y_contact*1e9,np.real(solution["J_interface"]),label="Re J interface")
        ax.set(xlabel="Distance under contact (nm)",ylabel="Interface current density (A/m²)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d/(prefix+"_current_crowding.png"),dpi=240)
        plt.close(fig)
    return d
