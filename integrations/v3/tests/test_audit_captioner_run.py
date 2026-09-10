"""Read-only audit preserves observation/action timing and partial-run status."""
import json
from pathlib import Path
import tempfile
from unittest import TestCase

from integrations.v3.audit_captioner_run import audit, percentile


class AuditTests(TestCase):
    def test_interpolated_percentiles(self):
        self.assertEqual(percentile([4000, 1000], .5), 2500)
        self.assertEqual(percentile([4000, 1000], .95), 3850)
        self.assertIsNone(percentile([], .5))

    def test_completion_precedes_the_returned_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace, log = Path(temporary) / "trace", Path(temporary) / "log"
            trace.write_text(json.dumps(dict(episode_id="1", step=0, action=1,
                position_before=[0, 0, 0], position_after=[.25, 0, 0],
                distance_to_goal_before=10., distance_to_goal_after=9.75,
                decision=dict(captioner_completed=True, captioner_raw_response="model completion",
                    completion_evidence_frame_ids=[1], captioner_error_mode="NONE"))) + "\n")
            log.write_text("ep=1 s=0 2500ms\nrank=0 [1/1] id=1 steps=1 success=0.000 spl=0.000\n")
            result, = audit(trace, log)
            self.assertEqual(result["completion_audit"][0]["path_so_far_m"], 0.)
            self.assertEqual(result["completion_audit"][0]["path_after_action_m"], .25)
            self.assertEqual(result["path_xz_m"], .25)
            self.assertEqual(result["state"], "finished")
            self.assertTrue(result["trace_matches_reported_steps"])
            self.assertEqual(result["latency_samples"], 1)
            self.assertEqual(result["step_p50_ms"], 2500)
            log.write_text("ep=1 s=0 2500ms\n")
            result, = audit(trace, log)
            self.assertEqual(result["state"], "partial")
            self.assertIsNone(result["trace_matches_reported_steps"])
