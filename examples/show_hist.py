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

# The raw per-bin counts are noisy (every voxel count wiggles up/down even in an
# otherwise monotonic region), so a naive extrema search on `hist` directly finds
# hundreds of spurious one-bin bumps. Smoothing first (simple moving average)
# keeps only the extrema that reflect real histogram structure (e.g. the
# background/soft-tissue/bone boundaries useful for picking transfer-function
# control points).
SMOOTHING_WINDOW = 9
kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
smoothed_hist = np.convolve(hist, kernel, mode="same")

diff = np.diff(smoothed_hist)
sign = np.sign(diff)
sign[sign == 0] = 1  # treat a flat step as a continuation of the previous slope
sign_changes = np.diff(sign)

local_maxima = np.flatnonzero(sign_changes < 0) + 1  # slope + -> - : peak
local_minima = np.flatnonzero(sign_changes > 0) + 1  # slope - -> + : valley

print(f"Local maxima (intensity, smoothed count): {len(local_maxima)} found")
for i in local_maxima:
    print(f"  {i:3d}  {smoothed_hist[i]:.1f}")

print(f"Local minima (intensity, smoothed count): {len(local_minima)} found")
for i in local_minima:
    print(f"  {i:3d}  {smoothed_hist[i]:.1f}")

plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", alpha=0.5, label="raw histogram")
plt.plot(bin_edges[:-1], smoothed_hist, color="black", linewidth=1, label="smoothed")
plt.scatter(local_maxima, smoothed_hist[local_maxima], color="red", zorder=5, label="local maxima")
plt.scatter(local_minima, smoothed_hist[local_minima], color="blue", zorder=5, label="local minima")
plt.xlabel("Voxel intensity")
plt.ylabel("Number of voxels")
plt.title("3D Volume Intensity Histogram")
plt.yscale("log")  # opsional, berguna karena background biasanya sangat dominan
plt.legend()
plt.grid(True)

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "skull_histogram.png"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Histogram saved to: {output_path}")
plt.show()