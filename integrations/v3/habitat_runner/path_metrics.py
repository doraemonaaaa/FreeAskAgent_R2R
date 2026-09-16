"""Path-fidelity metrics of VLN-CE: nDTW and SDTW against the dense ground-truth path.

nDTW = exp(-DTW(agent path, reference path) / (|reference path| * success distance))

* reference path: the dense ``locations`` of VLN-CE's ``{split}_gt.json.gz``
  (the shortest-path follower's positions from start to goal), stored beside
  the split's episodes;
* agent path: the start position plus every position that differs from the
  previous one (turns in place add nothing), as VLN-CE records it;
* DTW: fastdtw with Euclidean point distance, VLN-CE's default (FDTW=True);
  exact DTW when fastdtw is not installed;
* SDTW = success * nDTW.
"""

import gzip
import json
from functools import lru_cache

import numpy as np

from . import settings

try:
    from fastdtw import fastdtw as _fastdtw
except ImportError:  # exact DTW is never larger, so nDTW is slightly higher
    _fastdtw = None

DTW_METHOD = "fastdtw" if _fastdtw is not None else "exact"


def gt_path_file(split):
    return settings.HABITAT_DATA / "datasets/vln/mp3d/r2r/v1" / split / "{}_gt.json.gz".format(split)


@lru_cache(maxsize=4)
def load_gt_locations(split):
    """episode_id -> dense reference locations, or None when the split has no gt file."""
    path = gt_path_file(split)
    if not path.is_file():
        return None
    with gzip.open(str(path), "rt") as handle:
        data = json.load(handle)
    return {str(key): value["locations"] for key, value in data.items()}


def _euclidean(a, b):
    return float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)))


def dtw_distance(path, reference):
    if _fastdtw is not None:
        return float(_fastdtw(path, reference, dist=_euclidean)[0])
    a = np.asarray(path, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    total = np.full((len(a) + 1, len(b) + 1), np.inf)
    total[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            total[i, j] = cost[i - 1, j - 1] + min(total[i - 1, j], total[i, j - 1], total[i - 1, j - 1])
    return float(total[-1, -1])


def dedup_locations(positions):
    """Agent path as VLN-CE records it: a position only when it changed."""
    locations = []
    for position in positions:
        point = [float(value) for value in position]
        if not locations or point != locations[-1]:
            locations.append(point)
    return locations


def ndtw(positions, reference, success_distance=None):
    distance = settings.SUCCESS_DISTANCE_M if success_distance is None else float(success_distance)
    return float(np.exp(-dtw_distance(dedup_locations(positions), reference) / (len(reference) * distance)))
