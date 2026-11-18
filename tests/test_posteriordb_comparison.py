"""Guard coordinate orientation and source-derived likelihood adjustments."""

from itertools import product
import unittest

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from scripts.posteriordb_comparison.runner import _unpack
from scripts.posteriordb_comparison.special import map_parameters


class ComparisonMappingsTest(unittest.TestCase):
    def test_stan_column_major_names_recover_logical_matrix(self):
        values = _unpack(
            ["M.1.1", "M.2.1", "M.1.2", "M.2.2", "scale"],
            [2.0, 5.0, 3.0, 7.0, 0.4],
        )
        np.testing.assert_array_equal(values["M"], [[2, 3], [5, 7]])
        self.assertEqual(values["scale"].shape, ())
        self.assertEqual(values["scale"], 0.4)

    def test_dog_mapping_preserves_effects_and_accounts_for_changed_prior(self):
        rho = 0.35
        stan = {
            "sigma_logit_ab": np.array([0.7, 1.8]),
            "L_logit_ab": np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]]),
            "z": np.array([[0.4, -0.9], [1.2, 0.3]]),
        }
        mapped, adjustments, _ = map_parameters("dogs_nonhierarchical", {}, stan)
        factor = np.diag(stan["sigma_logit_ab"]) @ stan["L_logit_ab"]
        np.testing.assert_allclose(
            mapped["z"] @ factor.T, stan["z"] @ factor, atol=1e-14
        )
        expected = norm.logpdf(mapped["z"]).sum() - norm.logpdf(stan["z"]).sum()
        self.assertAlmostEqual(
            adjustments["changed_standard_normal_effect_prior"], expected
        )

    def test_irt_mapping_preserves_latent_means_not_binary_coefficient_prior(self):
        design = np.column_stack((np.ones(5), [0, 0, 0, 1, 1]))
        stan = {"lambda_adj": np.array([0.3, 1.1]), "beta_free": np.array([0.2, -0.4])}
        mapped, adjustments, _ = map_parameters(
            "2pl_latent_reg_irt", {"W": design}, stan
        )
        centered = design.copy()
        centered[:, 1] -= design[:, 1].mean()
        original_design = centered.copy()
        original_design[:, 1] /= 2 * design[:, 1].std(ddof=1)
        np.testing.assert_allclose(
            original_design @ stan["lambda_adj"], centered @ mapped["lambda_adj"]
        )
        # StudentT(3) log-density shape is -2*log(1+x*x/3).
        expected = -2 * (
            np.log1p(mapped["lambda_adj"] ** 2 / 3).sum()
            - np.log1p(stan["lambda_adj"] ** 2 / 3).sum()
        )
        self.assertAlmostEqual(
            adjustments["changed_student_t_coefficient_prior"], expected
        )
        self.assertNotEqual(expected, 0)
        self.assertAlmostEqual(mapped["beta"].sum(), 0)

    def test_hmm_correction_matches_exhaustive_hidden_paths(self):
        data = {"y": [0.2, -0.8, 1.3]}
        stan = {
            "pi1": np.array([0.3, 0.7]),
            "A": np.array([[0.8, 0.2], [0.35, 0.65]]),
            "mu": np.array([-0.4, 1.0]),
            "sigma": np.array([0.6, 1.4]),
        }
        emission = norm.pdf(np.asarray(data["y"])[:, None], stan["mu"], stan["sigma"])
        old, new = 0.0, 0.0
        for states in product(range(2), repeat=3):
            weight = stan["pi1"][states[0]]
            for time in (1, 2):
                weight *= (
                    stan["A"][states[time - 1], states[time]]
                    * emission[time, states[time]]
                )
            old += weight * emission[0].prod()
            new += weight * emission[0, states[0]]
        _, adjustments, _ = map_parameters("hmm_gaussian", data, stan)
        self.assertAlmostEqual(
            adjustments["changed_initial_emission_likelihood"], np.log(new / old)
        )

    def test_iohmm_correction_matches_both_transition_orientations(self):
        data = {"u": [[1.0], [-0.4], [0.8]], "y": [0.1, -0.2, 1.1]}
        stan = {
            "pi1": np.array([0.25, 0.75]),
            "w": np.array([[-0.7], [0.4]]),
            "b": np.array([[0.6], [-0.3]]),
            "sigma": np.array([0.7, 1.2]),
        }
        predictors = np.asarray(data["u"])
        p_second = expit(
            (predictors @ stan["w"].T)[:, 1] - (predictors @ stan["w"].T)[:, 0]
        )
        gates = np.column_stack((1 - p_second, p_second))
        emission = norm.pdf(
            np.asarray(data["y"])[:, None], predictors @ stan["b"].T, stan["sigma"]
        )
        old, new = 0.0, 0.0
        for states in product(range(2), repeat=3):
            base = stan["pi1"][states[0]] * np.prod(
                [emission[time, state] for time, state in enumerate(states)]
            )
            old += base * gates[1, states[0]] * gates[2, states[1]]
            new += base * gates[1, states[1]] * gates[2, states[2]]
        _, adjustments, _ = map_parameters("iohmm_reg", data, stan)
        self.assertAlmostEqual(
            adjustments["changed_transition_gating_likelihood"], np.log(new / old)
        )


if __name__ == "__main__":
    unittest.main()
