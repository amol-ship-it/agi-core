"""
Tests for ARC atomic primitives.

STRIPPED TO ZERO: All primitive tests removed during zero-strip.
Tests will be re-added as each primitive is justified and added back.
"""

import unittest


class TestAtomicPrimitivesPlaceholder(unittest.TestCase):
    """Placeholder — primitive tests will be added as primitives are added."""

    def test_no_primitives(self):
        """Verify zero primitives are registered."""
        from domains.arc.transformation_primitives import (
            build_atomic_primitives,
            build_parameterized_primitives,
        )
        from domains.arc.perception_primitives import build_perception_primitives

        self.assertEqual(len(build_atomic_primitives()), 0)
        self.assertEqual(len(build_parameterized_primitives()), 0)
        self.assertEqual(len(build_perception_primitives()), 0)


if __name__ == "__main__":
    unittest.main()
