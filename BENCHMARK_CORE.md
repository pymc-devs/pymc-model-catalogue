# Benchmark Core (26 models)

A curated subset of the catalogue chosen to give diverse coverage for
benchmarking and optimization research: small ↔ large, centered ↔ noncentered
hierarchical, dense linear algebra, `scan`-based time series and ODEs, GPs,
mixtures, spatial CAR, survival, wide gather-heavy survey models, and one
discrete-vars model so the logp-only codepath is exercised.

## Tiny / canonical baselines
1. `models/eight_schools_noncentered.py` — smallest hierarchical, noncentered reparameterization
2. `models/BEST.py` — two-group t-test, trivial graph
3. `models/GLM_poisson_regression.py` — small GLM baseline
4. `models/GLM_negative_binomial_regression.py` — overdispersed counts

## Hierarchical / multilevel
5. `models/multilevel_varying_intercept_slope_noncentered.py` — classic varying-slopes
6. `models/GLM_hierarchical_binomial_rat_tumor.py` — hierarchical binomial
7. `models/rugby_analytics.py` — medium hierarchical with team effects

## Linear-algebra heavy
8. `models/lkj_cholesky_cov_mvnormal.py` — LKJ + Cholesky path
9. `models/probabilistic_matrix_factorization.py` — large dense factorization
10. `models/bayesian_sem_workflow.py` — SEM, structured covariances

## Time series / scan
11. `models/stochastic_volatility.py` — large GRW, classic perf target
12. `models/ar2.py` — small AR baseline
13. `models/time_series_generative_graph_ar2.py` — explicit scan-built AR
14. `models/bayesian_var_ireland.py` — multivariate VAR (scan + linalg)
15. `models/euler_maruyama_linear_sde.py` — SDE via scan

## Gaussian processes
16. `models/gp_marginal_matern52.py` — small marginal GP
17. `models/gp_births_hsgp.py` — HSGP, medium
18. `models/malaria_hsgp.py` — larger HSGP with covariates

## Mixtures
19. `models/marginalized_gaussian_mixture_model.py` — marginalized mixture
20. `models/dirichlet_mixture_of_multinomials.py` — Dirichlet-multinomial mixture

## Survival / ODE / spatial / large real-data
21. `models/frailty_coxph.py` — survival with frailty
22. `models/ode_lotka_volterra_pytensor_scan.py` — ODE through scan
23. `models/nyc_bym_traffic.py` — CAR/BYM spatial, large
24. `models/excess_deaths.py` — larger real dataset, structural model

## Survey / categorical
25. `models/irt_survey_zerosum_categorical.py` — explanatory IRT: many *scaled* zero-sum
    effect tables gathered per respondent, a latent factor, stereotype-ordinal Categorical
    likelihood over an (N, G, K) logit array; sized as a numba compile-cost sentinel
    (the cost is superlinear in table count)

## Discrete free variables
26. `models_discrete/occupancy_crossbill.py` — exercises the logp-only path
