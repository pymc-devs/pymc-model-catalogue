"""Model: Recursive discrete choice at an amusement park.

The model observes boardings across 36 attractions over 80 ten-minute intervals
and conditions on park entries and exits. A scan tracks expected guest counts
by most recent ride, with one group for guests who have yet to board. At each
step, boarders move to their chosen ride group, and departures reduce all groups
proportionally.

Choice utility depends on ride duration, thrill, posted wait and walking time.
A beta-binomial models total boardings per interval. A Dirichlet-multinomial
allocates that total across attractions. The model generates its synthetic counts.

NUMBA benchmark results (PyMC 6.4.0+9.gaf565aa28, PyTensor
3.3.8+15.g69b385476, Numba 0.68.0, float64).
- Original: 1.78 ms/call, logp -7101.6594, gradient norm 3416.0526.
- Frozen: 1.74 ms/call, with matching logp and gradient.
"""

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytensor
import pytensor.tensor as pt


def build_model() -> tuple[pm.Model, dict]:
    n_intervals, n_attractions = 80, 36
    attractions = [f"ride_{i + 1}" for i in range(n_attractions)]
    coords = {
        "time": np.arange(n_intervals),
        "previous_attraction": ["entrance", *attractions],
        "attraction": attractions,
        "feature": ["duration", "thrill", "wait"],
    }

    rng = np.random.default_rng(9391)
    day = np.linspace(0, 1, n_intervals)
    entries = rng.poisson(12 + 72 * np.exp(-((day - 0.22) / 0.23) ** 2))
    entries[day > 0.85] = 0

    # Guests arrive earlier in the day. Departures increase toward closing.
    exits = np.empty_like(entries)
    guests = 0
    for interval, new_guests in enumerate(entries):
        guests += new_guests
        exit_rate = 1 if interval == n_intervals - 1 else 0.008 + 0.17 * day[interval] ** 4
        exits[interval] = rng.binomial(guests, exit_rate)
        guests -= exits[interval]

    duration = rng.uniform(2, 12, size=n_attractions)
    thrill = rng.uniform(1, 5, size=n_attractions)
    # Each ride has a reference wait. The same daily pattern scales them all.
    baseline_wait = rng.uniform(3, 22, size=n_attractions)
    posted_wait = baseline_wait * (0.6 + np.sin(np.pi * day[:, None]))
    attributes = np.stack([
        np.broadcast_to((duration - 7) / 5, posted_wait.shape),
        np.broadcast_to(thrill - 3, posted_wait.shape),
        (posted_wait - 15) / 10,
    ], axis=-1)

    locations = rng.uniform(0, 10, size=(n_attractions, 2))
    origins = np.vstack([[5, 0], locations])
    walk_minutes_data = np.linalg.norm(origins[:, None, :] - locations, axis=-1)

    with pm.Model(coords=coords) as model:
        # Park gate counts for each ten-minute interval.
        park_entries = pmd.Data("park_entries", entries, dims="time")
        park_exits = pmd.Data("park_exits", exits, dims="time")
        attraction_features = pmd.Data(
            "attraction_features", pm.floatX(attributes), dims=("time", "attraction", "feature"),
        )
        walk_minutes = pmd.Data(
            "walk_minutes", pm.floatX(walk_minutes_data), dims=("previous_attraction", "attraction"),
        )

        attraction_intercept_sigma = pmd.HalfNormal("attraction_intercept_sigma", sigma=0.5)
        attraction_intercept = pmd.ZeroSumNormal(
            "attraction_intercept", sigma=attraction_intercept_sigma, core_dims="attraction",
        )
        attraction_beta = pmd.Normal("attraction_beta", mu=0, sigma=0.35, dims="feature")
        walk_minutes_beta = pmd.Normal("walk_minutes_beta", mu=0, sigma=0.07)
        attraction_utility = attraction_intercept + pmd.dot(
            attraction_features, attraction_beta, dim="feature",
        )
        assert attraction_utility.dims == ("attraction", "time")

        walk_utility = walk_minutes_beta * walk_minutes
        assert walk_utility.dims == ("previous_attraction", "attraction")

        utility = attraction_utility + walk_utility
        choice_probabilities = pmd.math.softmax(utility, dim="attraction")

        # Probability a guest boards a ride during one interval.
        boarding_rate = pmd.Beta("boarding_rate", mu=0.2, nu=10)

        # The gate headcount includes this interval's entries and only earlier exits.
        attendance_before_exits = (park_entries - park_exits).cumsum("time") + park_exits
        # Share of guests still inside after this interval's park exits.
        stay_fraction = (attendance_before_exits - park_exits) / pmd.math.maximum(
            attendance_before_exits, 1,
        )

        def advance(
            park_entries, stay_fraction, choice_probabilities, guests_per_attraction, boarding_rate,
        ):
            # Counts are grouped by most recent ride. New guests go into group 0,
            # which holds guests who have yet to ride.
            guests_per_attraction = guests_per_attraction[0].inc(
                park_entries.astype(guests_per_attraction.dtype),
            )

            # Expected destination counts if everyone inside boarded a ride.
            weighted_choices = guests_per_attraction @ choice_probabilities

            # Guests who do not board keep their last-ride group. Boarders move
            # to the group for the ride they just chose.
            expected_boardings = boarding_rate * weighted_choices
            nonboarders = (1 - boarding_rate) * guests_per_attraction
            after_boarding = nonboarders[1:].inc(expected_boardings)

            # Apply the park exits proportionally across groups.
            guests_per_attraction = after_boarding * stay_fraction
            return guests_per_attraction, weighted_choices

        closing_population, weighted_choices = pytensor.scan(
            advance,
            sequences=[
                park_entries.values,
                stay_fraction.astype(choice_probabilities.dtype).values,
                choice_probabilities.transpose("time", "previous_attraction", "attraction").values,
            ],
            outputs_info=[pt.zeros(n_attractions + 1, dtype=choice_probabilities.dtype), None],
            non_sequences=[boarding_rate.values],
            strict=True,
            return_updates=False,
        )
        pmd.Deterministic(
            "closing_population", closing_population, dims=("time", "previous_attraction"),
        )
        weighted_choices = pmd.as_xtensor(weighted_choices, dims=("time", "attraction"))

        total_precision = pmd.LogNormal("total_precision", mu=np.log(100), sigma=1.0)
        total_boardings = pmd.BetaBinomial(
            "total_boardings", n=attendance_before_exits,
            alpha=total_precision * boarding_rate,
            beta=total_precision * (1 - boarding_rate),
            dims="time",
        )
        choice_precision = pmd.LogNormal("choice_precision", mu=np.log(100), sigma=1.0)
        destination_share = weighted_choices / pmd.math.maximum(attendance_before_exits, 1)
        # An empty park has zero boardings. Give its zero-count row valid concentrations.
        pm.DirichletMultinomial(
            "boardings_by_attraction",
            n=total_boardings.values,
            a=pmd.math.where(
                attendance_before_exits > 0,
                choice_precision * destination_share,
                1.0,
            ).values,
            dims=("time", "attraction"),
        )

    simulation_values = {
        "attraction_intercept_sigma": 0.4,
        "attraction_beta": [-0.1, 0.25, -0.6],
        "walk_minutes_beta": -0.1,
        "boarding_rate": 0.12,
        "total_precision": 200.0,
        "choice_precision": 300.0,
    }
    simulation_model = pm.do(
        model,
        {
            name: pmd.as_xtensor(pm.floatX(value), dims=model[name].dims)
            for name, value in simulation_values.items()
        },
    )
    predicted = pm.sample_prior_predictive(
        model=simulation_model,
        draws=1,
        var_names=["boardings_by_attraction"],
        random_seed=26719,
        return_inferencedata=False,
        compile_kwargs={"mode": "FAST_COMPILE"},
    )
    boarding_counts = predicted["boardings_by_attraction"][0]
    model = pm.observe(
        model,
        {
            "total_boardings": boarding_counts.sum(axis=-1),
            "boardings_by_attraction": boarding_counts,
        },
    )
    return model, model.initial_point()


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
