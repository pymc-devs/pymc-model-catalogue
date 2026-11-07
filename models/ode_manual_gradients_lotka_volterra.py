"""
Model: Lotka-Volterra predator-prey with manual-gradient custom Op
Source: pymc-examples/examples/ode_models/ODE_with_manual_gradients.ipynb,
        Section: "The probabilistic model"
Authors: Sanmitra Ghosh (original notebook); updated by Oriol Abril-Pla, Virgile
         Andreani, Chris Fonnesbeck
Description: Infers Lotka-Volterra ODE parameters (alpha, beta, gamma, delta,
    x(0), y(0), sigma) from the Hudson's Bay lynx-hare time series. The forward
    solve and its vector-sensitivity product (VSP) are provided by a custom
    PyTensor Op that wraps `scipy.integrate.odeint`, with a manually derived
    gradient obtained via forward sensitivity analysis (sensitivity equations
    augmented to the state ODE). The Op's `grad` method constructs a second
    `ODEGradop` that applies the cached VSP, avoiding double ODE solves.

Changes from original:
- Theano -> PyTensor (pytensor.graph.op.Op, pytensor.graph.basic.Apply,
  pt.as_tensor_variable). The Op API is otherwise preserved.
- Fixed ODEop.make_node output type: previously reused `x.type()` (the 6-D
  parameter vector type), which made the Op's output incorrectly shape (6,)
  instead of (n_states * len(times),) = (42,). Replaced with an explicit
  TensorType of shape (42,).
- Inlined LotkaVolterraModel, ODEop/ODEGradop, solveCached, state/numpy_vsp
  helpers, and data inside build_model() per catalogue rules.
- Removed sampling/plotting code.
- Added ip capture + initval-clearing boilerplate at end of build_model().
- Under NUMBA mode, PyTensor uses object mode for this Python Op.

Benchmark results:
- Original:  logp = -176.1439, grad norm = 207.7586, 15.2 us/call (100000 evals)
- Frozen:    logp = -176.1439, grad norm = 207.7586, 15.2 us/call (100000 evals)
"""

import numpy as np
import pymc as pm


def build_model():
    import pytensor.tensor as pt
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op
    from scipy.integrate import odeint

    n_states = 2
    n_odeparams = 4
    n_ivs = 2

    class LotkaVolterraModel:
        def __init__(self, n_states, n_odeparams, n_ivs, y0=None):
            self._n_states = n_states
            self._n_odeparams = n_odeparams
            self._n_ivs = n_ivs
            self._y0 = y0

        def simulate(self, parameters, times):
            return self._simulate(parameters, times, False)

        def simulate_with_sensitivities(self, parameters, times):
            return self._simulate(parameters, times, True)

        def _simulate(self, parameters, times, sensitivities):
            alpha, beta, gamma, delta, Xt0, Yt0 = (x for x in parameters)

            def r(y, t, p):
                X, Y = y
                dX_dt = alpha * X - beta * X * Y
                dY_dt = -gamma * Y + delta * X * Y
                return dX_dt, dY_dt

            if sensitivities:

                def jac(y):
                    X, Y = y
                    ret = np.zeros((self._n_states, self._n_states))
                    ret[0, 0] = alpha - beta * Y
                    ret[0, 1] = -beta * X
                    ret[1, 0] = delta * Y
                    ret[1, 1] = -gamma + delta * X
                    return ret

                def dfdp(y):
                    X, Y = y
                    ret = np.zeros(
                        (self._n_states, self._n_odeparams + self._n_ivs)
                    )
                    ret[0, 0] = X
                    ret[0, 1] = -X * Y
                    ret[1, 2] = -Y
                    ret[1, 3] = X * Y
                    return ret

                def rhs(y_and_dydp, t, p):
                    y = y_and_dydp[0 : self._n_states]
                    dydp = y_and_dydp[self._n_states :].reshape(
                        (self._n_states, self._n_odeparams + self._n_ivs)
                    )
                    dydt = r(y, t, p)
                    d_dydp_dt = np.matmul(jac(y), dydp) + dfdp(y)
                    return np.concatenate((dydt, d_dydp_dt.reshape(-1)))

                y0 = np.zeros((2 * (n_odeparams + n_ivs)) + n_states)
                y0[6] = 1.0  # d[X]/dXt0 at t==0
                y0[13] = 1.0  # d[Y]/dYt0 at t==0
                y0[0:n_states] = [Xt0, Yt0]
                result = odeint(rhs, y0, times, (parameters,), rtol=1e-6, atol=1e-5)
                values = result[:, 0 : self._n_states]
                dvalues_dp = result[:, self._n_states :].reshape(
                    (len(times), self._n_states, self._n_odeparams + self._n_ivs)
                )
                return values, dvalues_dp
            else:
                values = odeint(
                    r, [Xt0, Yt0], times, (parameters,), rtol=1e-6, atol=1e-5
                )
                return values

    ode_model = LotkaVolterraModel(n_states, n_odeparams, n_ivs)

    class solveCached:
        def __init__(self, times, n_params, n_outputs):
            self._times = times
            self._n_params = n_params
            self._n_outputs = n_outputs
            self._cachedParam = np.zeros(n_params)
            self._cachedSens = np.zeros((len(times), n_outputs, n_params))
            self._cachedState = np.zeros((len(times), n_outputs))

        def __call__(self, x):
            if np.all(x == self._cachedParam):
                state, sens = self._cachedState, self._cachedSens
            else:
                state, sens = ode_model.simulate_with_sensitivities(x, times)
            return state, sens

    times = np.arange(0, 21)  # number of measurement points
    cached_solver = solveCached(times, n_odeparams + n_ivs, n_states)

    def state(x):
        State, Sens = cached_solver(np.array(x, dtype=np.float64))
        cached_solver._cachedState = State
        cached_solver._cachedSens = Sens
        cached_solver._cachedParam = x
        return State.reshape((2 * len(State),))

    def numpy_vsp(x, g):
        numpy_sens = cached_solver(np.array(x, dtype=np.float64))[1].reshape(
            (n_states * len(times), len(x))
        )
        return numpy_sens.T.dot(g)

    class ODEGradop(Op):
        def __init__(self, numpy_vsp):
            self._numpy_vsp = numpy_vsp

        def make_node(self, x, g):
            x = pt.as_tensor_variable(x)
            g = pt.as_tensor_variable(g)
            return Apply(self, [x, g], [g.type()])

        def perform(self, node, inputs_storage, output_storage):
            x = inputs_storage[0]
            g = inputs_storage[1]
            out = output_storage[0]
            out[0] = self._numpy_vsp(x, g)

    class ODEop(Op):
        def __init__(self, state, numpy_vsp):
            self._state = state
            self._numpy_vsp = numpy_vsp

        def make_node(self, x):
            x = pt.as_tensor_variable(x)
            out = pt.TensorType("float64", shape=(n_states * len(times),))()
            return Apply(self, [x], [out])

        def perform(self, node, inputs_storage, output_storage):
            x = inputs_storage[0]
            out = output_storage[0]
            out[0] = self._state(x)

        def grad(self, inputs, output_grads):
            x = inputs[0]
            g = output_grads[0]
            grad_op = ODEGradop(self._numpy_vsp)
            grad_op_apply = grad_op(x, g)
            return [grad_op_apply]

    my_ODEop = ODEop(state, numpy_vsp)

    # Hudson's Bay Company lynx-hare data
    # fmt: off
    Lynx = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                     8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6])
    Hare = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                     27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7])
    # fmt: on
    Y = np.vstack((Hare, Lynx)).T

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=1, sigma=0.5)
        beta = pm.Normal("beta", mu=0.05, sigma=0.05)
        gamma = pm.Normal("gamma", mu=1, sigma=0.5)
        delta = pm.Normal("delta", mu=0.05, sigma=0.05)

        xt0 = pm.Lognormal("xto", mu=np.log(10), sigma=1)
        yt0 = pm.Lognormal("yto", mu=np.log(10), sigma=1)
        sigma = pm.Lognormal("sigma", mu=-1, sigma=1, shape=2)

        all_params = pm.math.stack([alpha, beta, gamma, delta, xt0, yt0], axis=0)
        ode_sol = my_ODEop(all_params)
        forward = ode_sol.reshape(Y.shape)

        Y_obs = pm.Lognormal("Y_obs", mu=pm.math.log(forward), sigma=sigma, observed=Y)

    ip = model.initial_point()
    model.rvs_to_initial_values = {rv: None for rv in model.free_RVs}
    return model, ip


if __name__ == "__main__":
    from _benchmark import run_benchmark

    run_benchmark(build_model)
