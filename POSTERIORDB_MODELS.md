# Posteriordb models

The catalogue includes all 148 model/data pairs in
[posteriordb](https://github.com/stan-dev/posteriordb) at
[revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4),
covering 121 model definitions and 91 datasets. Each pair has an
explicit `posteriordb_<model>_<dataset>.py` file under `models/` or
`models_discrete/`, with bundled source JSON archives.

See the [model index](#modeldata-pairs) for individual ports and
[`models/posteriordb/source/`](models/posteriordb/source/) for source references
and attribution.

## Using the models

```python
from models.posteriordb_blr_sblri import build_model

model, ip = build_model()

# The source namespace re-exports the same builders for convenience.
from models.posteriordb import blr_sblrc
other_model, other_ip = blr_sblrc()
```

The ARMA/GARCH ports accept same-length data updates through `observations`.
The MNIST ports accept new prediction batches through `x`; update the
`observation` coordinates when the batch length changes.

Run a pair with `python models/posteriordb_blr_sblri.py`.
See [benchmarking instructions](./BENCHMARKING.md) for selecting entries in ASV.

## Validation and intentional departures

Each model docstring reports how much the log probability changed between the
same two points in Stan and PyMC, where a Stan reference exists. It gives the
log probabilities at both points, the difference between the two changes, and
the tolerance used to decide whether they agree.

The recorded two-point outcomes are:

| Outcome | Model/data pairs |
| --- | ---: |
| Raw deltas agree | 139 |
| Agree only after explicit prior or likelihood adjustments | 5 |
| Numerical mismatch remains | 3 |
| No Stan reference (discrete M0) | 1 |

`bones_model` misses tolerance because `OrderedLogistic` subtracts nearly equal
CDF values. The `lotka_volterra` and `sir` discrepancies remain unexplained;
their Stan and PyMC ports use different ODE solvers.

Comparisons use constrained densities with automatic support-transform Jacobians
disabled, following
[A Rosetta stone for PyMC and Stan](https://gist.github.com/ricardoV94/c421ddacb3ba7a19ac46efa253ea466c).
Parameter-independent constants cancel in `delta = logp(B) - logp(A)`.
Residuals are Stan delta minus PyMC delta; adjustments are added to Stan logp.
Two points do not establish global equivalence, and agreement after a prior or
likelihood adjustment does not make the original targets equivalent.

Measurements used BridgeStan 2.9.0 (Stan 2.39.0), PyMC 6.4.0 and PyTensor 3.3.7.
The tolerance is `1e-6 + 1e-8 * max(abs(delta_Stan), abs(delta_PyMC))`, using the
adjusted Stan delta where applicable.

The following ports deliberately follow the stated model where it differs from
the published implementation. Their docstrings briefly describe the departures.
Some changes affect priors: the dog-effects covariance changes their joint prior,
and IRT covariate scaling changes the induced coefficient priors.
Comparisons must account for those changes explicitly, using a corresponding Stan
reference or documented density adjustments.

| Models | Effect | Intentional departure | Evidence |
| --- | --- | --- | --- |
| `hmm_gaussian` | Likelihood | Use the first observation's state-specific emission density. | [Original HMM definition](https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L239-L271) |
| `iohmm_reg` | Likelihood | Use destination-state gating weights. Native Gaussian mixtures eliminate the conditionally independent states. | [Original description and generator](https://github.com/luisdamiano/stancon18/blob/master/main.Rmd#L840-L875) |
| `dogs_nonhierarchical` | Prior | Use the declared covariance `diag(sigma) @ Omega @ diag(sigma)`. | [PyMC implementation](models/posteriordb_dogs_nonhierarchical_dogs.py) |
| `2pl_latent_reg_irt`, `gpcm_latent_reg_irt` | Induced prior | Scale binary covariates by their range, not twice their sample standard deviation. | [Furr's model description, sections 1.1 and 1.3](https://mc-stan.org/learn-stan/case-studies/rasch_and_2pl.html) |

The shared scaling correction also covers `grsm_latent_reg_irt`. Its bundled intercept-only dataset does not change.

## Model/data pairs

| Upstream posterior ID | Catalogue file |
| --- | --- |
| `GLMM_Poisson_data-GLMM_Poisson_model` | [models/posteriordb_GLMM_Poisson_model_GLMM_Poisson_data.py](models/posteriordb_GLMM_Poisson_model_GLMM_Poisson_data.py) |
| `GLMM_data-GLMM1_model` | [models/posteriordb_GLMM1_model_GLMM_data.py](models/posteriordb_GLMM1_model_GLMM_data.py) |
| `GLM_Binomial_data-GLM_Binomial_model` | [models/posteriordb_GLM_Binomial_model_GLM_Binomial_data.py](models/posteriordb_GLM_Binomial_model_GLM_Binomial_data.py) |
| `GLM_Poisson_Data-GLM_Poisson_model` | [models/posteriordb_GLM_Poisson_model_GLM_Poisson_Data.py](models/posteriordb_GLM_Poisson_model_GLM_Poisson_Data.py) |
| `M0_data-M0_model` | [models/posteriordb_M0_model_M0_data.py](models/posteriordb_M0_model_M0_data.py) |
| `M0_data-M0_model_discrete` | [models_discrete/posteriordb_M0_model_discrete_M0_data.py](models_discrete/posteriordb_M0_model_discrete_M0_data.py) |
| `Mb_data-Mb_model` | [models/posteriordb_Mb_model_Mb_data.py](models/posteriordb_Mb_model_Mb_data.py) |
| `Mh_data-Mh_model` | [models/posteriordb_Mh_model_Mh_data.py](models/posteriordb_Mh_model_Mh_data.py) |
| `Mt_data-Mt_model` | [models/posteriordb_Mt_model_Mt_data.py](models/posteriordb_Mt_model_Mt_data.py) |
| `Mtbh_data-Mtbh_model` | [models/posteriordb_Mtbh_model_Mtbh_data.py](models/posteriordb_Mtbh_model_Mtbh_data.py) |
| `Mth_data-Mth_model` | [models/posteriordb_Mth_model_Mth_data.py](models/posteriordb_Mth_model_Mth_data.py) |
| `Rate_1_data-Rate_1_model` | [models/posteriordb_Rate_1_model_Rate_1_data.py](models/posteriordb_Rate_1_model_Rate_1_data.py) |
| `Rate_2_data-Rate_2_model` | [models/posteriordb_Rate_2_model_Rate_2_data.py](models/posteriordb_Rate_2_model_Rate_2_data.py) |
| `Rate_3_data-Rate_3_model` | [models/posteriordb_Rate_3_model_Rate_3_data.py](models/posteriordb_Rate_3_model_Rate_3_data.py) |
| `Rate_4_data-Rate_4_model` | [models/posteriordb_Rate_4_model_Rate_4_data.py](models/posteriordb_Rate_4_model_Rate_4_data.py) |
| `Rate_5_data-Rate_5_model` | [models/posteriordb_Rate_5_model_Rate_5_data.py](models/posteriordb_Rate_5_model_Rate_5_data.py) |
| `Survey_data-Survey_model` | [models/posteriordb_Survey_model_Survey_data.py](models/posteriordb_Survey_model_Survey_data.py) |
| `arK-arK` | [models/posteriordb_arK_arK.py](models/posteriordb_arK_arK.py) |
| `arma-arma11` | [models/posteriordb_arma11_arma.py](models/posteriordb_arma11_arma.py) |
| `bball_drive_event_0-hmm_drive_0` | [models/posteriordb_hmm_drive_0_bball_drive_event_0.py](models/posteriordb_hmm_drive_0_bball_drive_event_0.py) |
| `bball_drive_event_1-hmm_drive_1` | [models/posteriordb_hmm_drive_1_bball_drive_event_1.py](models/posteriordb_hmm_drive_1_bball_drive_event_1.py) |
| `bones_data-bones_model` | [models/posteriordb_bones_model_bones_data.py](models/posteriordb_bones_model_bones_data.py) |
| `butterfly-multi_occupancy` | [models/posteriordb_multi_occupancy_butterfly.py](models/posteriordb_multi_occupancy_butterfly.py) |
| `diamonds-diamonds` | [models/posteriordb_diamonds_diamonds.py](models/posteriordb_diamonds_diamonds.py) |
| `dogs-dogs` | [models/posteriordb_dogs_dogs.py](models/posteriordb_dogs_dogs.py) |
| `dogs-dogs_hierarchical` | [models/posteriordb_dogs_hierarchical_dogs.py](models/posteriordb_dogs_hierarchical_dogs.py) |
| `dogs-dogs_log` | [models/posteriordb_dogs_log_dogs.py](models/posteriordb_dogs_log_dogs.py) |
| `dogs-dogs_nonhierarchical` | [models/posteriordb_dogs_nonhierarchical_dogs.py](models/posteriordb_dogs_nonhierarchical_dogs.py) |
| `dugongs_data-dugongs_model` | [models/posteriordb_dugongs_model_dugongs_data.py](models/posteriordb_dugongs_model_dugongs_data.py) |
| `earnings-earn_height` | [models/posteriordb_earn_height_earnings.py](models/posteriordb_earn_height_earnings.py) |
| `earnings-log10earn_height` | [models/posteriordb_log10earn_height_earnings.py](models/posteriordb_log10earn_height_earnings.py) |
| `earnings-logearn_height` | [models/posteriordb_logearn_height_earnings.py](models/posteriordb_logearn_height_earnings.py) |
| `earnings-logearn_height_male` | [models/posteriordb_logearn_height_male_earnings.py](models/posteriordb_logearn_height_male_earnings.py) |
| `earnings-logearn_interaction` | [models/posteriordb_logearn_interaction_earnings.py](models/posteriordb_logearn_interaction_earnings.py) |
| `earnings-logearn_interaction_z` | [models/posteriordb_logearn_interaction_z_earnings.py](models/posteriordb_logearn_interaction_z_earnings.py) |
| `earnings-logearn_logheight_male` | [models/posteriordb_logearn_logheight_male_earnings.py](models/posteriordb_logearn_logheight_male_earnings.py) |
| `ecdc0401-covid19imperial_v2` | [models/posteriordb_covid19imperial_v2_ecdc0401.py](models/posteriordb_covid19imperial_v2_ecdc0401.py) |
| `ecdc0401-covid19imperial_v3` | [models/posteriordb_covid19imperial_v3_ecdc0401.py](models/posteriordb_covid19imperial_v3_ecdc0401.py) |
| `ecdc0501-covid19imperial_v2` | [models/posteriordb_covid19imperial_v2_ecdc0501.py](models/posteriordb_covid19imperial_v2_ecdc0501.py) |
| `ecdc0501-covid19imperial_v3` | [models/posteriordb_covid19imperial_v3_ecdc0501.py](models/posteriordb_covid19imperial_v3_ecdc0501.py) |
| `eight_schools-eight_schools_centered` | [models/posteriordb_eight_schools_centered_eight_schools.py](models/posteriordb_eight_schools_centered_eight_schools.py) |
| `eight_schools-eight_schools_noncentered` | [models/posteriordb_eight_schools_noncentered_eight_schools.py](models/posteriordb_eight_schools_noncentered_eight_schools.py) |
| `election88-election88_full` | [models/posteriordb_election88_full_election88.py](models/posteriordb_election88_full_election88.py) |
| `fims_Aus_Jpn_irt-2pl_latent_reg_irt` | [models/posteriordb_model_2pl_latent_reg_irt_fims_Aus_Jpn_irt.py](models/posteriordb_model_2pl_latent_reg_irt_fims_Aus_Jpn_irt.py) |
| `garch-garch11` | [models/posteriordb_garch11_garch.py](models/posteriordb_garch11_garch.py) |
| `gp_pois_regr-gp_pois_regr` | [models/posteriordb_gp_pois_regr_gp_pois_regr.py](models/posteriordb_gp_pois_regr_gp_pois_regr.py) |
| `gp_pois_regr-gp_regr` | [models/posteriordb_gp_regr_gp_pois_regr.py](models/posteriordb_gp_regr_gp_pois_regr.py) |
| `hmm_example-hmm_example` | [models/posteriordb_hmm_example_hmm_example.py](models/posteriordb_hmm_example_hmm_example.py) |
| `hmm_gaussian_simulated-hmm_gaussian` | [models/posteriordb_hmm_gaussian_hmm_gaussian_simulated.py](models/posteriordb_hmm_gaussian_hmm_gaussian_simulated.py) |
| `hudson_lynx_hare-lotka_volterra` | [models/posteriordb_lotka_volterra_hudson_lynx_hare.py](models/posteriordb_lotka_volterra_hudson_lynx_hare.py) |
| `iohmm_reg_simulated-iohmm_reg` | [models/posteriordb_iohmm_reg_iohmm_reg_simulated.py](models/posteriordb_iohmm_reg_iohmm_reg_simulated.py) |
| `irt_2pl-irt_2pl` | [models/posteriordb_irt_2pl_irt_2pl.py](models/posteriordb_irt_2pl_irt_2pl.py) |
| `kidiq-kidscore_interaction` | [models/posteriordb_kidscore_interaction_kidiq.py](models/posteriordb_kidscore_interaction_kidiq.py) |
| `kidiq-kidscore_momhs` | [models/posteriordb_kidscore_momhs_kidiq.py](models/posteriordb_kidscore_momhs_kidiq.py) |
| `kidiq-kidscore_momhsiq` | [models/posteriordb_kidscore_momhsiq_kidiq.py](models/posteriordb_kidscore_momhsiq_kidiq.py) |
| `kidiq-kidscore_momiq` | [models/posteriordb_kidscore_momiq_kidiq.py](models/posteriordb_kidscore_momiq_kidiq.py) |
| `kidiq_with_mom_work-kidscore_interaction_c` | [models/posteriordb_kidscore_interaction_c_kidiq_with_mom_work.py](models/posteriordb_kidscore_interaction_c_kidiq_with_mom_work.py) |
| `kidiq_with_mom_work-kidscore_interaction_c2` | [models/posteriordb_kidscore_interaction_c2_kidiq_with_mom_work.py](models/posteriordb_kidscore_interaction_c2_kidiq_with_mom_work.py) |
| `kidiq_with_mom_work-kidscore_interaction_z` | [models/posteriordb_kidscore_interaction_z_kidiq_with_mom_work.py](models/posteriordb_kidscore_interaction_z_kidiq_with_mom_work.py) |
| `kidiq_with_mom_work-kidscore_mom_work` | [models/posteriordb_kidscore_mom_work_kidiq_with_mom_work.py](models/posteriordb_kidscore_mom_work_kidiq_with_mom_work.py) |
| `kilpisjarvi_mod-kilpisjarvi` | [models/posteriordb_kilpisjarvi_kilpisjarvi_mod.py](models/posteriordb_kilpisjarvi_kilpisjarvi_mod.py) |
| `loss_curves-losscurve_sislob` | [models/posteriordb_losscurve_sislob_loss_curves.py](models/posteriordb_losscurve_sislob_loss_curves.py) |
| `low_dim_gauss_mix-low_dim_gauss_mix` | [models/posteriordb_low_dim_gauss_mix_low_dim_gauss_mix.py](models/posteriordb_low_dim_gauss_mix_low_dim_gauss_mix.py) |
| `low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse` | [models/posteriordb_low_dim_gauss_mix_collapse_low_dim_gauss_mix_collapse.py](models/posteriordb_low_dim_gauss_mix_collapse_low_dim_gauss_mix_collapse.py) |
| `lsat_data-lsat_model` | [models/posteriordb_lsat_model_lsat_data.py](models/posteriordb_lsat_model_lsat_data.py) |
| `mcycle_gp-accel_gp` | [models/posteriordb_accel_gp_mcycle_gp.py](models/posteriordb_accel_gp_mcycle_gp.py) |
| `mcycle_splines-accel_splines` | [models/posteriordb_accel_splines_mcycle_splines.py](models/posteriordb_accel_splines_mcycle_splines.py) |
| `mesquite-logmesquite` | [models/posteriordb_logmesquite_mesquite.py](models/posteriordb_logmesquite_mesquite.py) |
| `mesquite-logmesquite_logva` | [models/posteriordb_logmesquite_logva_mesquite.py](models/posteriordb_logmesquite_logva_mesquite.py) |
| `mesquite-logmesquite_logvas` | [models/posteriordb_logmesquite_logvas_mesquite.py](models/posteriordb_logmesquite_logvas_mesquite.py) |
| `mesquite-logmesquite_logvash` | [models/posteriordb_logmesquite_logvash_mesquite.py](models/posteriordb_logmesquite_logvash_mesquite.py) |
| `mesquite-logmesquite_logvolume` | [models/posteriordb_logmesquite_logvolume_mesquite.py](models/posteriordb_logmesquite_logvolume_mesquite.py) |
| `mesquite-mesquite` | [models/posteriordb_mesquite_mesquite.py](models/posteriordb_mesquite_mesquite.py) |
| `mnist-nn_rbm1bJ100` | [models/posteriordb_nn_rbm1bJ100_mnist.py](models/posteriordb_nn_rbm1bJ100_mnist.py) |
| `mnist_100-nn_rbm1bJ10` | [models/posteriordb_nn_rbm1bJ10_mnist_100.py](models/posteriordb_nn_rbm1bJ10_mnist_100.py) |
| `nes1972-nes` | [models/posteriordb_nes_nes1972.py](models/posteriordb_nes_nes1972.py) |
| `nes1976-nes` | [models/posteriordb_nes_nes1976.py](models/posteriordb_nes_nes1976.py) |
| `nes1980-nes` | [models/posteriordb_nes_nes1980.py](models/posteriordb_nes_nes1980.py) |
| `nes1984-nes` | [models/posteriordb_nes_nes1984.py](models/posteriordb_nes_nes1984.py) |
| `nes1988-nes` | [models/posteriordb_nes_nes1988.py](models/posteriordb_nes_nes1988.py) |
| `nes1992-nes` | [models/posteriordb_nes_nes1992.py](models/posteriordb_nes_nes1992.py) |
| `nes1996-nes` | [models/posteriordb_nes_nes1996.py](models/posteriordb_nes_nes1996.py) |
| `nes2000-nes` | [models/posteriordb_nes_nes2000.py](models/posteriordb_nes_nes2000.py) |
| `nes_logit_data-nes_logit_model` | [models/posteriordb_nes_logit_model_nes_logit_data.py](models/posteriordb_nes_logit_model_nes_logit_data.py) |
| `normal_2-normal_mixture` | [models/posteriordb_normal_mixture_normal_2.py](models/posteriordb_normal_mixture_normal_2.py) |
| `normal_5-normal_mixture_k` | [models/posteriordb_normal_mixture_k_normal_5.py](models/posteriordb_normal_mixture_k_normal_5.py) |
| `one_comp_mm_elim_abs-one_comp_mm_elim_abs` | [models/posteriordb_one_comp_mm_elim_abs_one_comp_mm_elim_abs.py](models/posteriordb_one_comp_mm_elim_abs_one_comp_mm_elim_abs.py) |
| `ovarian-logistic_regression_rhs` | [models/posteriordb_logistic_regression_rhs_ovarian.py](models/posteriordb_logistic_regression_rhs_ovarian.py) |
| `pilots-pilots` | [models/posteriordb_pilots_pilots.py](models/posteriordb_pilots_pilots.py) |
| `prideprejudice_chapter-ldaK5` | [models/posteriordb_ldaK5_prideprejudice_chapter.py](models/posteriordb_ldaK5_prideprejudice_chapter.py) |
| `prideprejudice_paragraph-ldaK5` | [models/posteriordb_ldaK5_prideprejudice_paragraph.py](models/posteriordb_ldaK5_prideprejudice_paragraph.py) |
| `prostate-logistic_regression_rhs` | [models/posteriordb_logistic_regression_rhs_prostate.py](models/posteriordb_logistic_regression_rhs_prostate.py) |
| `radon_all-radon_county_intercept` | [models/posteriordb_radon_county_intercept_radon_all.py](models/posteriordb_radon_county_intercept_radon_all.py) |
| `radon_all-radon_hierarchical_intercept_centered` | [models/posteriordb_radon_hierarchical_intercept_centered_radon_all.py](models/posteriordb_radon_hierarchical_intercept_centered_radon_all.py) |
| `radon_all-radon_hierarchical_intercept_noncentered` | [models/posteriordb_radon_hierarchical_intercept_noncentered_radon_all.py](models/posteriordb_radon_hierarchical_intercept_noncentered_radon_all.py) |
| `radon_all-radon_partially_pooled_centered` | [models/posteriordb_radon_partially_pooled_centered_radon_all.py](models/posteriordb_radon_partially_pooled_centered_radon_all.py) |
| `radon_all-radon_partially_pooled_noncentered` | [models/posteriordb_radon_partially_pooled_noncentered_radon_all.py](models/posteriordb_radon_partially_pooled_noncentered_radon_all.py) |
| `radon_all-radon_pooled` | [models/posteriordb_radon_pooled_radon_all.py](models/posteriordb_radon_pooled_radon_all.py) |
| `radon_all-radon_variable_intercept_centered` | [models/posteriordb_radon_variable_intercept_centered_radon_all.py](models/posteriordb_radon_variable_intercept_centered_radon_all.py) |
| `radon_all-radon_variable_intercept_noncentered` | [models/posteriordb_radon_variable_intercept_noncentered_radon_all.py](models/posteriordb_radon_variable_intercept_noncentered_radon_all.py) |
| `radon_all-radon_variable_intercept_slope_centered` | [models/posteriordb_radon_variable_intercept_slope_centered_radon_all.py](models/posteriordb_radon_variable_intercept_slope_centered_radon_all.py) |
| `radon_all-radon_variable_intercept_slope_noncentered` | [models/posteriordb_radon_variable_intercept_slope_noncentered_radon_all.py](models/posteriordb_radon_variable_intercept_slope_noncentered_radon_all.py) |
| `radon_all-radon_variable_slope_centered` | [models/posteriordb_radon_variable_slope_centered_radon_all.py](models/posteriordb_radon_variable_slope_centered_radon_all.py) |
| `radon_all-radon_variable_slope_noncentered` | [models/posteriordb_radon_variable_slope_noncentered_radon_all.py](models/posteriordb_radon_variable_slope_noncentered_radon_all.py) |
| `radon_mn-radon_county_intercept` | [models/posteriordb_radon_county_intercept_radon_mn.py](models/posteriordb_radon_county_intercept_radon_mn.py) |
| `radon_mn-radon_hierarchical_intercept_centered` | [models/posteriordb_radon_hierarchical_intercept_centered_radon_mn.py](models/posteriordb_radon_hierarchical_intercept_centered_radon_mn.py) |
| `radon_mn-radon_hierarchical_intercept_noncentered` | [models/posteriordb_radon_hierarchical_intercept_noncentered_radon_mn.py](models/posteriordb_radon_hierarchical_intercept_noncentered_radon_mn.py) |
| `radon_mn-radon_partially_pooled_centered` | [models/posteriordb_radon_partially_pooled_centered_radon_mn.py](models/posteriordb_radon_partially_pooled_centered_radon_mn.py) |
| `radon_mn-radon_partially_pooled_noncentered` | [models/posteriordb_radon_partially_pooled_noncentered_radon_mn.py](models/posteriordb_radon_partially_pooled_noncentered_radon_mn.py) |
| `radon_mn-radon_pooled` | [models/posteriordb_radon_pooled_radon_mn.py](models/posteriordb_radon_pooled_radon_mn.py) |
| `radon_mn-radon_variable_intercept_centered` | [models/posteriordb_radon_variable_intercept_centered_radon_mn.py](models/posteriordb_radon_variable_intercept_centered_radon_mn.py) |
| `radon_mn-radon_variable_intercept_noncentered` | [models/posteriordb_radon_variable_intercept_noncentered_radon_mn.py](models/posteriordb_radon_variable_intercept_noncentered_radon_mn.py) |
| `radon_mn-radon_variable_intercept_slope_centered` | [models/posteriordb_radon_variable_intercept_slope_centered_radon_mn.py](models/posteriordb_radon_variable_intercept_slope_centered_radon_mn.py) |
| `radon_mn-radon_variable_intercept_slope_noncentered` | [models/posteriordb_radon_variable_intercept_slope_noncentered_radon_mn.py](models/posteriordb_radon_variable_intercept_slope_noncentered_radon_mn.py) |
| `radon_mn-radon_variable_slope_centered` | [models/posteriordb_radon_variable_slope_centered_radon_mn.py](models/posteriordb_radon_variable_slope_centered_radon_mn.py) |
| `radon_mn-radon_variable_slope_noncentered` | [models/posteriordb_radon_variable_slope_noncentered_radon_mn.py](models/posteriordb_radon_variable_slope_noncentered_radon_mn.py) |
| `radon_mod-radon_county` | [models/posteriordb_radon_county_radon_mod.py](models/posteriordb_radon_county_radon_mod.py) |
| `rats_data-rats_model` | [models/posteriordb_rats_model_rats_data.py](models/posteriordb_rats_model_rats_data.py) |
| `rstan_downloads-prophet` | [models/posteriordb_prophet_rstan_downloads.py](models/posteriordb_prophet_rstan_downloads.py) |
| `sat-hier_2pl` | [models/posteriordb_hier_2pl_sat.py](models/posteriordb_hier_2pl_sat.py) |
| `sblrc-blr` | [models/posteriordb_blr_sblrc.py](models/posteriordb_blr_sblrc.py) |
| `sblri-blr` | [models/posteriordb_blr_sblri.py](models/posteriordb_blr_sblri.py) |
| `science_irt-grsm_latent_reg_irt` | [models/posteriordb_grsm_latent_reg_irt_science_irt.py](models/posteriordb_grsm_latent_reg_irt_science_irt.py) |
| `seeds_data-seeds_centered_model` | [models/posteriordb_seeds_centered_model_seeds_data.py](models/posteriordb_seeds_centered_model_seeds_data.py) |
| `seeds_data-seeds_model` | [models/posteriordb_seeds_model_seeds_data.py](models/posteriordb_seeds_model_seeds_data.py) |
| `seeds_data-seeds_stanified_model` | [models/posteriordb_seeds_stanified_model_seeds_data.py](models/posteriordb_seeds_stanified_model_seeds_data.py) |
| `sesame_data-sesame_one_pred_a` | [models/posteriordb_sesame_one_pred_a_sesame_data.py](models/posteriordb_sesame_one_pred_a_sesame_data.py) |
| `sir-sir` | [models/posteriordb_sir_sir.py](models/posteriordb_sir_sir.py) |
| `soil_carbon-soil_incubation` | [models/posteriordb_soil_incubation_soil_carbon.py](models/posteriordb_soil_incubation_soil_carbon.py) |
| `state_wide_presidential_votes-hierarchical_gp` | [models/posteriordb_hierarchical_gp_state_wide_presidential_votes.py](models/posteriordb_hierarchical_gp_state_wide_presidential_votes.py) |
| `surgical_data-surgical_model` | [models/posteriordb_surgical_model_surgical_data.py](models/posteriordb_surgical_model_surgical_data.py) |
| `synthetic_grid_RBF_kernels-kronecker_gp` | [models/posteriordb_kronecker_gp_synthetic_grid_RBF_kernels.py](models/posteriordb_kronecker_gp_synthetic_grid_RBF_kernels.py) |
| `three_docs1200-ldaK2` | [models/posteriordb_ldaK2_three_docs1200.py](models/posteriordb_ldaK2_three_docs1200.py) |
| `three_men1-ldaK2` | [models/posteriordb_ldaK2_three_men1.py](models/posteriordb_ldaK2_three_men1.py) |
| `three_men2-ldaK2` | [models/posteriordb_ldaK2_three_men2.py](models/posteriordb_ldaK2_three_men2.py) |
| `three_men3-ldaK2` | [models/posteriordb_ldaK2_three_men3.py](models/posteriordb_ldaK2_three_men3.py) |
| `timssAusTwn_irt-gpcm_latent_reg_irt` | [models/posteriordb_gpcm_latent_reg_irt_timssAusTwn_irt.py](models/posteriordb_gpcm_latent_reg_irt_timssAusTwn_irt.py) |
| `traffic_accident_nyc-bym2_offset_only` | [models/posteriordb_bym2_offset_only_traffic_accident_nyc.py](models/posteriordb_bym2_offset_only_traffic_accident_nyc.py) |
| `uk_drivers-state_space_stochastic_level_stochastic_seasonal` | [models/posteriordb_state_space_stochastic_level_stochastic_seasonal_uk_drivers.py](models/posteriordb_state_space_stochastic_level_stochastic_seasonal_uk_drivers.py) |
| `wells_data-wells_daae_c_model` | [models/posteriordb_wells_daae_c_model_wells_data.py](models/posteriordb_wells_daae_c_model_wells_data.py) |
| `wells_data-wells_dae_c_model` | [models/posteriordb_wells_dae_c_model_wells_data.py](models/posteriordb_wells_dae_c_model_wells_data.py) |
| `wells_data-wells_dae_inter_model` | [models/posteriordb_wells_dae_inter_model_wells_data.py](models/posteriordb_wells_dae_inter_model_wells_data.py) |
| `wells_data-wells_dae_model` | [models/posteriordb_wells_dae_model_wells_data.py](models/posteriordb_wells_dae_model_wells_data.py) |
| `wells_data-wells_dist` | [models/posteriordb_wells_dist_wells_data.py](models/posteriordb_wells_dist_wells_data.py) |
| `wells_data-wells_dist100_model` | [models/posteriordb_wells_dist100_model_wells_data.py](models/posteriordb_wells_dist100_model_wells_data.py) |
| `wells_data-wells_dist100ars_model` | [models/posteriordb_wells_dist100ars_model_wells_data.py](models/posteriordb_wells_dist100ars_model_wells_data.py) |
| `wells_data-wells_interaction_c_model` | [models/posteriordb_wells_interaction_c_model_wells_data.py](models/posteriordb_wells_interaction_c_model_wells_data.py) |
| `wells_data-wells_interaction_model` | [models/posteriordb_wells_interaction_model_wells_data.py](models/posteriordb_wells_interaction_model_wells_data.py) |
