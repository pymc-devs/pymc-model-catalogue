# Adding model examples

Each catalogue file represents one model/data pair and defines a no-argument
`build_model()` returning `(model, initial_point)`. Put continuous-latent models
in `models/` and models with discrete free variables in `models_discrete/`.
Discrete observations alone do not require the logp-only directory.

Preserve the source model's priors, likelihood, parameterization, `pm.Data`,
dimensions, deterministics, and initial values. Keep construction, data loading,
and supporting functions inside the builder. Omit sampling and plotting.

Bundle data locally and preserve the source format when practical. Small arrays
can be inlined. Share data files between models that use the same dataset.

Each file's docstring should identify:

- The model, authors, description, and model/data identity.
- A link to the source model. For posteriordb, the shared index records the revision and dataset.
- Changes from the source, with equations or parameter mappings where needed.
  Explicitly identify prior changes, including changes induced by scaling or
  reparameterization, and explain why they were made.
- For Stan ports, the two-point logp results and any model-specific adjustments,
  following the [comparison protocol](./POSTERIORDB_MODELS.md#validation-and-intentional-departures).
  If no comparison is available, say why.
- Benchmark results and the environment used for those measurements.

Capture the initial point before clearing initial-value overrides:

```python
ip = model.initial_point()
model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
return model, ip
```

Use the shared `run_benchmark` helper under `if __name__ == "__main__":`, passing
`discrete=True` for the logp-only path. Check finite logp and, where applicable,
gradients in NUMBA mode for original and frozen graphs at the same point.
