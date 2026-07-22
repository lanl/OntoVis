import os

import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "vis_male_128x256x256_uint8.raw"
)

volume = np.fromfile(
    DATASET_PATH,
    dtype=np.uint8
).reshape((128, 256, 256))

hist, bin_edges = np.histogram(
    volume,
    bins=256,
    range=(0, 256)
)

plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge")
plt.xlabel("Voxel intensity")
plt.ylabel("Number of voxels")
plt.title("3D Volume Intensity Histogram")
plt.yscale("log")  # opsional, berguna karena background biasanya sangat dominan
plt.grid(True)

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "skull_histogram.png"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Histogram saved to: {output_path}")
plt.show()