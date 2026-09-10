"""The actual IPC worker receives the recording run's evidence directory."""
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from integrations.v3.run_habitat import WaypointActorProcess


class EvidenceEnvironmentTests(TestCase):
    def test_intermediate_observation_transports_its_own_depth_and_calibration(self):
        import numpy as np
        actor = object.__new__(WaypointActorProcess)
        actor._png = lambda value: "rgb"
        actor._array = lambda value: value.tolist()
        actor._request = lambda request: request
        depth = np.full((2, 3), 2.)
        k = np.eye(3)
        result = actor.observe(np.zeros((2, 3, 3)), np.eye(4), depth=depth, intrinsics=k)
        self.assertEqual(result["depth"], depth.tolist())
        self.assertEqual(result["intrinsics"], k.tolist())
        self.assertNotIn("depth", actor.observe(np.zeros((2, 3, 3)), np.eye(4)))

    @patch("integrations.v3.run_habitat.subprocess.Popen")
    def test_recording_directory_is_absolute_and_not_a_parent_environment_write(self, popen):
        with patch.dict("os.environ", {}, clear=True):
            WaypointActorProcess("python", "worker.py", "remote-model", evidence_dir="outputs/run/evidence")
            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["JOYAI_EVIDENCE_DIR"], str(Path("outputs/run/evidence").resolve()))
            import os
            self.assertNotIn("JOYAI_EVIDENCE_DIR", os.environ)

    @patch("integrations.v3.run_habitat.subprocess.Popen")
    def test_explicit_location_or_opt_out_is_preserved(self, popen):
        for configured in ("/tmp/caller-selected-evidence", ""):
            with self.subTest(configured=configured), patch.dict("os.environ", {"JOYAI_EVIDENCE_DIR": configured}, clear=True):
                WaypointActorProcess("python", "worker.py", "remote-model", evidence_dir="outputs/run/evidence")
                self.assertEqual(popen.call_args.kwargs["env"]["JOYAI_EVIDENCE_DIR"], configured)

    @patch("integrations.v3.run_habitat.subprocess.Popen")
    def test_non_recording_legacy_caller_does_not_enable_archiving(self, popen):
        with patch.dict("os.environ", {}, clear=True):
            WaypointActorProcess("python", "worker.py", "remote-model")
            self.assertNotIn("JOYAI_EVIDENCE_DIR", popen.call_args.kwargs["env"])
