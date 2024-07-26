"""Ordinary Differential Equation dynamics models"""

import typing
import scipy
import safe_autonomy_simulation.dynamics.dynamics as d

try:
    import jax.numpy as np
    import jax.experimental.ode as jode
    JAX_AVAILABLE = True
except ImportError:
    import numpy as np
    JAX_AVAILABLE = False


class ODEDynamics(d.Dynamics):
    """
    State transition implementation for generic Ordinary Differential Equation dynamics models.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    trajectory_samples : int, optional
        Number of trajectory samples to generate and step through, by default 0
    state_min : float or np.ndarray, optional
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -np.inf
    state_max : float or np.ndarray, optional
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, np.inf
    state_dot_min : float or np.ndarray, optional
        Minimum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -inf
    state_dot_max : float or np.ndarray, optional
        Maximum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, +inf
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'Euler'].
        'RK45' is slow but very accurate. If jax is available, can be JIT compiled for speed.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    """

    def __init__(
        self,
        trajectory_samples: int = 0,
        state_min: typing.Union[float, np.ndarray] = -np.inf,
        state_max: typing.Union[float, np.ndarray] = np.inf,
        state_dot_min: typing.Union[float, np.ndarray] = -np.inf,
        state_dot_max: typing.Union[float, np.ndarray] = np.inf,
        integration_method: str = "RK45",
    ):
        super().__init__(
            state_min=state_min,
            state_max=state_max,
        )

        assert integration_method in [
            "RK45",
            "Euler",
        ], f"invalid integration method {integration_method}, must be one of 'RK45', 'Euler'"
        self.integration_method = integration_method
        self.state_dot_min = state_dot_min
        self.state_dot_max = state_dot_max

        assert isinstance(
            trajectory_samples, int
        ), f"trajectory_samples must be an integer, got {trajectory_samples}"
        assert (
            trajectory_samples >= 0
        ), f"trajectory_samples must be non-negative, got {trajectory_samples}"
        self.trajectory_samples = trajectory_samples

        self.trajectory = None
        self.trajectory_t = None

    def compute_state_dot(
        self, t: float, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """
        Computes the instantaneous time derivative of the state vector

        Parameters
        ----------
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        state : np.ndarray
            Current state vector at time t.
        control : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            Instantaneous time derivative of the state vector.
        """
        state_dot = self._compute_state_dot(t, state, control)
        state_dot = self._clip_state_dot_direct(state_dot)
        state_dot = self._clip_state_dot_by_state_limits(state, state_dot)
        return state_dot

    def _compute_state_dot(
        self, t: float, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def _clip_state_dot_direct(self, state_dot: np.ndarray):
        """Clips state derivative values to be within `self.state_dot_min` and `self.state_dot_max`

        Parameters
        ----------
        state_dot : np.ndarray
            State derivative values

        Returns
        -------
        np.ndarray
            State derivative values clipped to be within state_dot_min and state_dot_max
        """
        return self.np.clip(state_dot, self.state_dot_min, self.state_dot_max)

    def _clip_state_dot_by_state_limits(self, state: np.ndarray, state_dot: np.ndarray):
        """Clips state derivative values where the state is at its limits

        State derivative values are clipped such that they do not push the state
        beyond the limits defined by `self.state_min` and `self.state_max`. This
        is done by clipping the state derivative values to be non-negative when
        the state is at its lower limit, and non-positive when the state is at
        its upper limit.

        Parameters
        ----------
        state : np.ndarray
            Current state vector
        state_dot : np.ndarray
            State derivative values

        Returns
        -------
        np.ndarray
            State derivative values clipped where the state is at its limits
        """
        lower_bounded_states = state <= self.state_min
        upper_bounded_states = state >= self.state_max

        lower_bounded_clipped = self.np.clip(state_dot, 0, np.inf)
        upper_bounded_clipped = self.np.clip(state_dot, -np.inf, 0)

        state_dot = self.np.where(
            lower_bounded_states, lower_bounded_clipped, state_dot
        )
        state_dot = self.np.where(
            upper_bounded_states, upper_bounded_clipped, state_dot
        )

        return state_dot

    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray):
        if self.integration_method == "RK45":
            if not JAX_AVAILABLE:
                t_eval = None
                if self.trajectory_samples > 0:
                    t_eval = np.linspace(0, step_size, self.trajectory_samples + 1)[1:]

                sol = scipy.integrate.solve_ivp(
                    self.compute_state_dot,
                    (0, step_size),
                    state,
                    args=(control,),
                    t_eval=t_eval,
                )

                self.trajectory = sol.y.T
                self.trajectory_t = sol.t

                next_state = sol.y[:, -1]  # save last timestep of integration solution
                state_dot = self.compute_state_dot(step_size, next_state, control)
            else:
                assert (
                    self.trajectory_samples <= 0
                ), "trajectory sampling not currently supported with rk45 jax integration"

                sol = jode.odeint(  # pylint: disable=used-before-assignment
                    self.compute_state_dot_jax,
                    state,
                    np.linspace(0.0, step_size, 11),
                    control,
                )
                next_state = sol[-1, :]  # save last timestep of integration solution
                state_dot = self.compute_state_dot(step_size, next_state, control)
        elif self.integration_method == "Euler":
            assert (
                self.trajectory_samples <= 0
            ), "trajectory sampling not currently supported with euler integration"
            state_dot = self.compute_state_dot(0, state, control)
            next_state = state + step_size * state_dot
        else:
            raise ValueError(f"invalid integration method '{self.integration_method}'")

        return next_state, state_dot

    def compute_state_dot_jax(self, state: np.ndarray, t: float, control: np.ndarray):
        """Compute state dot for jax odeint

        Jax requires a specific signature for the function passed to odeint. This function
        is a wrapper around the compute_state_dot function that has the correct signature.

        Parameters
        ----------
        state : np.ndarray
            Current state vector at time t.
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        control : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            Instantaneous time derivative of the state vector.
        """
        return self._compute_state_dot(t, state, control)


class ControlAffineODEDynamics(ODEDynamics):
    """
    State transition implementation for control affine Ordinary Differential Equation dynamics models of the form
        dx/dt = f(x) + g(x)u.

    At Each point in the numerical integration processes, f(x) and g(x) are computed at the integration point

    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    state_min : float or np.ndarray, optional
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -np.inf
    state_max : float or np.ndarray, optional
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, np.inf
    state_dot_min : float or np.ndarray, optional
        Minimum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -inf
    state_dot_max : float or np.ndarray, optional
        Maximum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, +inf
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'Euler'].
        'RK45' is slow but very accurate. If jax is available, can be JIT compiled for speed.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    """

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray):
        state_dot = (
            self.state_transition_system(state)
            + self.state_transition_input(state) @ control
        )
        return state_dot

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        """Computes the system state contribution to the system state's time derivative

        i.e. implements f(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : np.ndarray
            Current state vector of the system.

        Returns
        -------
        np.ndarray
            state time derivative contribution from the current system state
        """
        raise NotImplementedError

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        """Computes the control input matrix contribution to the system state's time derivative

        i.e. implements g(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : np.ndarray
            Current state vector of the system.

        Returns
        -------
        np.ndarray
            input matrix in state space representation time derivative
        """
        raise NotImplementedError


class LinearODEDynamics(ControlAffineODEDynamics):
    """
    State transition implementation for generic Linear Ordinary Differential Equation dynamics models of the form
    dx/dt = Ax+Bu.
    Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    A : np.ndarray
        State transition matrix. A of dx/dt = Ax + Bu. Should be dimension len(n) x len(n)
    B : np.ndarray
        Control input matrix. B of dx/dt = Ax + Bu. Should be dimension len(n) x len(u)
    trajectory_samples : int, optional
        number of trajectory samples the generate and store on steps, by default 0
    state_min : float or np.ndarray, optional
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -np.inf
    state_max : float or np.ndarray, optional
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, np.inf
    state_dot_min : float or np.ndarray, optional
        Minimum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, -inf
    state_dot_max : float or np.ndarray, optional
        Maximum allowable value for the state time derivative. State derivative values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
        By default, +inf
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'Euler'].
        'RK45' is slow but very accurate. If jax is available, can be JIT compiled for speed.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        trajectory_samples: int = 0,
        state_min: typing.Union[float, np.ndarray] = -np.inf,
        state_max: typing.Union[float, np.ndarray] = np.inf,
        state_dot_min: typing.Union[float, np.ndarray] = -np.inf,
        state_dot_max: typing.Union[float, np.ndarray] = np.inf,
        integration_method: str = "RK45",
    ):
        super().__init__(
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            integration_method=integration_method,
        )

        assert len(A.shape) == 2, f"A must be a 2D matrix. Instead got shape {A.shape}"
        assert len(B.shape) == 2, f"B must be a 2D matrix. Instead got shape {B.shape}"
        assert (
            A.shape[0] == A.shape[1]
        ), f"A must be a square matrix, not shape {A.shape}"
        assert A.shape[1] == B.shape[0], (
            "number of columns in A must match the number of rows in B."
            + f" However, got shapes {A.shape} for A and {B.shape} for B"
        )

        self.A = self.np.copy(A)
        self.B = self.np.copy(B)

    def state_transition_system(self, state: np.ndarray) -> np.ndarray:
        return self.A @ state

    def state_transition_input(self, state: np.ndarray) -> np.ndarray:
        return self.B
