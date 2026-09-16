
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from conquest2a.density import (
    bandden,
    plot_density,
)  # to process BandStructure.dat
from conquest2a.conquest import *  # necessary, (1)
conquest_species_map = conquest_species({1: "K", 2: "Cu", 3: "Cu", 4: "F"})
import scienceplots

plt.style.use(["science", "no-latex"])

mpl.rcParams.update(
    {
        #     # LaTeX rendering
        #     "text.usetex":          False,
        #     "font.family":          "serif",
        #     "font.serif":           ["Computer Modern Roman"],
        #     "mathtext.fontset":     "cm",
        #     # Font sizes
        "font.size": 12,
        #     "axes.titlesize":       12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        #     "legend.fontsize":      10,
        #     # Axes & spines
        #     "axes.linewidth":       0.8,
        #     "axes.edgecolor":       "#333333",
        #     "axes.facecolor":       "white",
        #     "axes.spines.top":      False,
        #     "axes.spines.right":    False,
        #     # Muted grey gridlines
        "axes.grid": False,
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "axes.axisbelow": True,  # grid behind data
        # Flat / frameless legend
        "legend.frameon": True,
        "legend.framealpha": 1,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.loc": "best",
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        #     # Lines & markers
        "lines.linewidth": 1.25,
        "lines.markersize": 5,
        #     # Figure
        #     "figure.dpi":           150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        #     "savefig.pad_inches":   0.05,
        #     "figure.figsize":       (3.5, 2.8),   # single-column width (inches)
    }
)


fig, (ax_val, ax_cond) = plt.subplots(1, 2, figsize=(8, 5))

path_to_dir = r"./bandden" + "/"
den1 = bandden(
    directory=path_to_dir,
    hkl=[-9.937, -1, 61.7977],
    offset=3.5,
    operations="+++",
    bands=[87, 88],
    units="ang",
)

den2 = bandden(
    directory=path_to_dir,
    hkl=[-2.42693, -3.6017, -1],
    offset=-2.21346,
    operations="+++",
    bands=[89, 90],
    units="ang",
)
valence = plot_density(den1, show_atoms=True)
#
_, _, im1 = valence.run(
    "test.png",
    ax=ax_val,
    shift=(8.8, -5.9),
    normalise=True,
    thickness=1,
    vmax=None,
    rotation = 40.0,
    xlabel=r"$x^\prime  (\mathrm{\AA})$",
    ylabel=r"$z^\prime  (\mathrm{\AA})$",
    xlim=(-3,3),
    ylim=(-3, 3),
    window_repeat=3,
    atom_repeat=2,
    grid_points=300,
    save=False,
    show_colorbar=False,
    label_atoms=False,
    atom_edgecolor={"Mn": "#000000", "O": "#000000"},
    atom_size=50,
    atom_kwargs={"Mn": {"linewidth": 1}, "O": {"linestyle": "--", "linewidth": 0.7}},
    atom_bgcolor={"Mn": "#00000000", "O": "#00000000"},
    aspect="equal",
)

conduction = plot_density(den2, show_atoms=True)
_, _, im2 = conduction.run(
    "test_cond.png",
    ax=ax_cond,
    save=False,
    normalise=True,
    thickness=1,
    shift=(-2.3, 4.2),
    #auto_orient=True,
    xlabel=r"$y^\prime  (\mathrm{\AA})$",
    ylabel=r"$x^\prime (\mathrm{\AA})$",
    show_ticks=True,
    show_colorbar=False,
    atom_symbols=["Mn", "O"],
    vmax=None,
    window_repeat=2,
    atom_repeat=1,
    grid_points=300,
    xlim=(-3, 3),
    ylim=(-3, 3),
    label_atoms=False,
    atom_edgecolor={"Mn": "#000000", "O": "#000000"},
    atom_size=50,
    atom_bgcolor={"Mn": "#00000000", "O": "#00000000"},
    atom_kwargs={"Mn": {"linewidth": 1}, "O": {"linestyle": "--", "linewidth": 0.7}},
)


ax_val.text(-2.7, 2.5, "(a)")
ax_cond.text(-2.7, 2.5, "(b)")
fig.tight_layout(rect=(0, 0.1, 1, 1))  # leave a strip at the bottom for the colorbar
cax = fig.add_axes(
    [0.15, 0.06, 0.7, 0.03]
)  # [left, bottom, width, height] in figure fraction
cbar = fig.colorbar(im1, cax=cax, orientation="horizontal")
cbar_label = r"Normalised total band density"
cbar.set_label(cbar_label, fontsize=12)
plt.savefig("lamno3_banddens.svg")