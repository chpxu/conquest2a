"""
This file plots a band structure from a Conquest band structure output file using matplotlib.

Run in root of repo:
    python3 examples/plot_test_band.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
from conquest2a.band import bst_processor

# Adjust the paths below to point to your actual files
data_dir = Path(__file__).parent.parent.resolve() / "tests"
bst_file = data_dir / "test_BandStructure.dat"

bst_processor_instance = bst_processor(bst_file=bst_file)
fig = plt.figure(figsize=(8, 6))
first_1 = 1
first_2 = 1
for band in bst_processor_instance.bands:
    if band.spin == 1:
        plt.plot(
            band.kpoint,
            band.energies,
            color="red",
            linewidth=0.5,
            label="Spin up" if first_1 == 1 else "",
        )
        first_1 = 0

    else:
        plt.plot(
            band.kpoint,
            band.energies,
            color="black",
            linewidth=0.5,
            label="Spin down" if first_2 == 1 else "",
        )
        first_2 = 0

plt.legend()
plt.xlabel("k-point")
plt.xticks(ticks=[1, 20, 40, 60, 80, 96], labels=["Γ", "X", "U", "T", "Y", "Γ"])
plt.ylabel("Energy (eV)")
plt.savefig(data_dir / "test_band_structure.png", dpi=600, bbox_inches="tight")
