"""Check posterior data ownership and constrained model initialization."""

import unittest

from models.posteriordb import load_data


class PosteriorDataSelectionTest(unittest.TestCase):
    def test_rejects_a_posterior_that_belongs_to_another_model(self):
        with self.assertRaises(ValueError):
            load_data("eight_schools_centered", "nes1972-nes")


class PosteriorInitializationTest(unittest.TestCase):
    def test_positive_ordered_scales_have_finite_initial_density(self):
        import numpy as np

        from models.posteriordb.state_space_stochastic_level_stochastic_seasonal import (
            build_model,
        )

        model, point = build_model()
        self.assertTrue(all(np.isfinite(value).all() for value in point.values()))
        scale_logp = model.compile_logp(vars=[model["sigma"]])
        self.assertTrue(np.isfinite(scale_logp(point)))


if __name__ == "__main__":
    unittest.main()
