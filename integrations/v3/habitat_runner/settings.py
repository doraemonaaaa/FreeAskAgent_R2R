"""Paths, quiet logging and the runtime geometry every runner module shares.

``TURN_ANGLE_DEG`` / ``FORWARD_STEP_M`` / ``SENSOR_HEIGHT_M`` / ``CAMERA`` are
module state: ``configure()`` rebinds them from the config once in ``main()``
and the other modules read them at call time (``settings.CAMERA``).
"""

import logging
import os
import sys
import warnings
from pathlib import Path

for _name in ("habitat", "habitat_sim", "magnum", "corrade", "transformers", "torch"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.captureWarnings(True)
warnings.filterwarnings("ignore")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

ROOT = Path(__file__).resolve().parents[3]
HABITAT_ROOT = ROOT.parent / "habitat" / "habitat-lab"
HABITAT_DATA = ROOT.parent / "habitat" / "data"
AGENTFLOW_ROOT = ROOT.parent / "FreeAskAgent"
for _path in (str(ROOT), str(HABITAT_ROOT / "habitat-lab")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from integrations.v3.camera_model import CameraModel  # noqa: E402

# The actor asks for turns in degrees and this runner executes whole repeats of
# the simulator's turn primitive, so this value is half of a contract with the
# actor's TURN_STEP_DEG rather than a private simulator setting. A mismatch
# would silently round every requested turn down.
TURN_ANGLE_DEG = 15  # config.yaml robot.turn_angle_deg
FORWARD_STEP_M = 0.25  # config.yaml robot.forward_step_m
# Height of both cameras above the agent's base. The actor needs the same
# number to tell floor pixels from wall pixels when it snaps a waypoint, so it
# is defined once (sensor_config.yaml camera.height_m) and passed through.
SENSOR_HEIGHT_M = 1.25
# R2R-CE success radius; oracle success uses the same radius.
SUCCESS_DISTANCE_M = 3.0
# Nav camera model (intrinsics / distortion / mount).
CAMERA = CameraModel(width=640, height=480, hfov_deg=90.0, height_m=SENSOR_HEIGHT_M)


def configure(*, camera, turn_angle_deg, forward_step_m):
    """Install the config's geometry for every module of the runner."""
    global CAMERA, TURN_ANGLE_DEG, FORWARD_STEP_M, SENSOR_HEIGHT_M
    CAMERA = camera
    TURN_ANGLE_DEG = int(turn_angle_deg)
    FORWARD_STEP_M = float(forward_step_m)
    SENSOR_HEIGHT_M = float(camera.height_m)
