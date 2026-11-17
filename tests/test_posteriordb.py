"""Check constrained model initialization."""

import unittest


class PosteriorInitializationTest(unittest.TestCase):
    def test_positive_ordered_scales_have_finite_initial_density(self):
        import numpy as np

        from models.posteriordb_state_space_stochastic_level_stochastic_seasonal_uk_drivers import (
            build_model,
        )

        model, point = build_model()
        self.assertTrue(all(np.isfinite(value).all() for value in point.values()))
        scale_logp = model.compile_logp(vars=[model["sigma"]])
        self.assertTrue(np.isfinite(scale_logp(point)))


if __name__ == "__main__":
    unittest.main()
