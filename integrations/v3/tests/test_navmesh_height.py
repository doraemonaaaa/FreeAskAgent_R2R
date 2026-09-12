"""Run with Habitat's Python: python -m unittest integrations.v3.tests.test_navmesh_height."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from integrations.v3.habitat_runner.actor_process import WaypointActorProcess
from integrations.v3.habitat_runner.sensors import navigable_window
from integrations.v3.vln_waypoint_worker import _decode_array


class NavmeshHeightTests(TestCase):
    def test_optional_height_grid_does_not_change_traversability(self):
        pathfinder = SimpleNamespace(
            nav_mesh_settings=SimpleNamespace(cell_height=.2),
            is_navigable=lambda xyz, tolerance: xyz[0] >= 0,
            snap_point=Mock(side_effect=lambda xyz: [xyz[0], .25, xyz[2]]),
        )
        env = SimpleNamespace(sim=SimpleNamespace(pathfinder=pathfinder,
            get_agent_state=lambda: SimpleNamespace(position=(0., .15, 0.))))
        old = navigable_window(env, radius_m=.5, resolution_m=.5)
        pathfinder.snap_point.assert_not_called()
        self.assertNotIn("height_m", old)
        new = navigable_window(env, radius_m=.5, resolution_m=.5, include_heights=True)
        np.testing.assert_array_equal(old["mask"], new["mask"])
        self.assertTrue(np.isnan(new["height_m"][:, 0]).all())
        np.testing.assert_allclose(new["height_m"][:, 1], .25)
        self.assertEqual(new["height_cell_m"], .2)
        self.assertEqual(pathfinder.snap_point.call_count, int(new["mask"].sum()))

    def test_height_grid_survives_ipc_without_starting_worker(self):
        actor = object.__new__(WaypointActorProcess)
        actor._request = Mock(return_value={"stop": True})
        heights = np.array([[.15, np.nan], [.3, .45]], dtype=np.float32)
        window = dict(origin_xz=(0, 0), resolution_m=.25,
                      mask=np.ones((2, 2), dtype=bool), height_m=heights, height_cell_m=.2)
        actor.act(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4)), "route",
                  np.eye(3), np.eye(4), navigable=window)
        payload = actor._request.call_args[0][0]["navigable"]
        np.testing.assert_allclose(_decode_array(payload["height_m"]), heights, equal_nan=True)
        self.assertEqual(payload["height_cell_m"], .2)
        del window["height_m"]
        actor.act(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4)), "route",
                  np.eye(3), np.eye(4), navigable=window)
        self.assertNotIn("height_m", actor._request.call_args[0][0]["navigable"])
