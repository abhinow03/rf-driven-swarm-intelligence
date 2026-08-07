"""Unit tests for scripts/verify_upstream_physics.py's pattern-matching logic (isolated from
the real repo's current fix/unfixed state) plus one integration check against the ACTUAL
current source, which is expected to fail until both upstream fixes are pulled locally --
see docs/V5_STATE.json / docs/V5_LOG.md for why."""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.verify_upstream_physics import (  # noqa: E402
    _SHARED_BRANCH_RE, check_geometry_fix, check_acceleration_fix,
)


class TestSharedBranchPattern(unittest.TestCase):
    def test_matches_or_form(self):
        snippet = 'elif formation_type == "dispersed" or formation_type == "converging":'
        self.assertIsNotNone(_SHARED_BRANCH_RE.search(snippet))

    def test_matches_tuple_membership_form(self):
        snippet = 'elif formation_type in ("dispersed", "converging"):'
        self.assertIsNotNone(_SHARED_BRANCH_RE.search(snippet))

    def test_matches_reversed_tuple_order(self):
        snippet = 'elif formation_type in ("converging", "dispersed"):'
        self.assertIsNotNone(_SHARED_BRANCH_RE.search(snippet))

    def test_does_not_match_separate_branches(self):
        snippet = (
            'elif formation_type == "dispersed":\n'
            '    offsets = rng.uniform(low=[-30, -30, -15], high=[30, 30, 15], size=(6, 3))\n'
            'elif formation_type == "converging":\n'
            '    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)\n'
        )
        self.assertIsNone(_SHARED_BRANCH_RE.search(snippet))

    def test_does_not_match_unrelated_code(self):
        snippet = 'elif formation_type == "shield":\n    offsets = np.array([[-10, 10, 5]])\n'
        self.assertIsNone(_SHARED_BRANCH_RE.search(snippet))


class TestCurrentRepoStateIsUnfixed(unittest.TestCase):
    """Integration check against the real, current src/swarm_intent/ source. Both of these
    are EXPECTED TO FAIL right now (that is the whole point of the guard) -- once the
    teammate's upstream fixes (commit 9158b081, see V5_LOG.md) are pulled into this repo,
    both assertions below should be updated to assertIsNone (no exception)."""

    def test_geometry_fix_not_yet_present(self):
        with self.assertRaises(AssertionError):
            check_geometry_fix()

    def test_acceleration_fix_not_yet_present(self):
        with self.assertRaises(AssertionError):
            check_acceleration_fix()


if __name__ == "__main__":
    unittest.main()
