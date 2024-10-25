"""Ordinary Differential Equation dynamics models"""

import typing
import safe_autonomy_simulation
import safe_autonomy_simulation.dynamics as dynamics
import numpy as np
import jax.numpy as jnp


class ODEDynamics(dynamics.Dynamics):
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
    use_jax : bool, optional
        EXPERIMENTAL: Use JAX to accelerate state transition computation, by default False.
    """

    def __init__(
        self,
        trajectory_samples: int = 0,
        state_min: typing.Union[float, np.ndarray] = -np.inf,
        state_max: typing.Union[float, np.ndarray] = np.inf,
        state_dot_min: typing.Union[float, np.ndarray] = -np.inf,
        state_dot_max: typing.Union[float, np.ndarray] = np.inf,
        integration_method: str = "RK45",
        use_jax: bool = False,
    ):
        super().__init__(
            state_min=state_min,
            state_max=state_max,
            use_jax=use_jax,
        )

        assert (
            integration_method
            in [
                "RK45",
                "Euler",
            ]
        ), f"invalid integration method {integration_method}, must be one of 'RK45', 'Euler'"
        self.integration_method = integration_method
        self.state_dot_min = self.np.copy(state_dot_min)
        self.state_dot_max = self.np.copy(state_dot_max)

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
        self,
        t: float,
        state: np.ndarray | jnp.ndarray,
        control: np.ndarray | jnp.ndarray,
    ) -> np.ndarray | jnp.ndarray:
        """
        Computes the instantaneous time derivative of the state vector

        Parameters
        ----------
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        state : np.ndarray | jnp.ndarray
            Current state vector at time t.
        control : np.ndarray | jnp.ndarray
            Control vector.

        Returns
        -------
        np.ndarray | jnp.ndarray
            Instantaneous time derivative of the state vector.
        """
        # Compute state derivative
        state_dot = self._compute_state_dot(t, state, control)

        # Select clip function
        clip_at_state_limits_fn = (
            safe_autonomy_simulation.dynamics.utils.clip_state_dot_at_state_limits
            if not self.use_jax
            else safe_autonomy_simulation.jax.ode.clip_state_dot_at_state_limits
        )

        # Clip state derivative values
        state_dot = np.clip(state_dot, self.state_dot_min, self.state_dot_max)
        
        # Clip state derivative values to ensure state remains within bounds
        state_dot = clip_at_state_limits_fn(
            state=state,
            state_dot=state_dot,
            s_min=self.state_min,
            s_max=self.state_max,
        )

        return state_dot

    def _compute_state_dot(
        self,
        t: float,
        state: np.ndarray | jnp.ndarray,
        control: np.ndarray | jnp.ndarray,
    ) -> np.ndarray | jnp.ndarray:
        raise NotImplementedError

    def _step(
        self,
        step_size: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        if not self.use_jax:  # use numpy arrays and functions
            step_fn = (
                safe_autonomy_simulation.dynamics.utils.step_rk45
                if self.integration_method == "RK45"
                else safe_autonomy_simulation.dynamics.utils.step_euler
            )
        else:  # use jax arrays and JIT functions
            step_fn = (
                safe_autonomy_simulation.jax.ode.step_rk45
                if self.integration_method == "RK45"
                else safe_autonomy_simulation.jax.ode.step_euler
            )
            state = jnp.array(state)
            control = jnp.array(control)

        next_state, state_dot, self.trajectory, self.trajectory_t = step_fn(
            f=self.compute_state_dot,
            step_size=step_size,
            state=state,
            control=control,
        )

        if self.use_jax:  # cast back to numpy
            next_state = np.array(next_state)
            state_dot = np.array(state_dot)

        return next_state, state_dot


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

    def _compute_state_dot(
        self,
        t: float,
        state: np.ndarray | jnp.ndarray,
        control: np.ndarray | jnp.ndarray,
    ) -> np.ndarray | jnp.ndarray:
        transition_fn = (
            safe_autonomy_simulation.dynamics.utils.affine_transition
            if not self.use_jax
            else safe_autonomy_simulation.jax.ode.affine_transition
        )
        state_dot = transition_fn(
            state=state,
            control=control,
            f=self.state_transition_system,
            g=self.state_transition_input,
        )
        return state_dot

    def state_transition_system(
        self, state: np.ndarray | jnp.ndarray
    ) -> np.ndarray | jnp.ndarray:
        """Computes the system state contribution to the system state's time derivative

        i.e. implements f(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : np.ndarray | jnp.ndarray
            Current state vector of the system.

        Returns
        -------
        np.ndarray | jnp.ndarray
            state time derivative contribution from the current system state
        """
        raise NotImplementedError

    def state_transition_input(
        self, state: np.ndarray | jnp.ndarray
    ) -> np.ndarray | jnp.ndarray:
        """Computes the control input matrix contribution to the system state's time derivative

        i.e. implements g(x) from dx/dt = f(x) + g(x)u

        Parameters
        ----------
        state : np.ndarray | jnp.ndarray
            Current state vector of the system.

        Returns
        -------
        np.ndarray | jnp.ndarray
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
    use_jax : bool, optional
        EXPERIMENTAL: Use JAX to accelerate state transition computation, by default False.
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
        use_jax: bool = False,
    ):
        super().__init__(
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            integration_method=integration_method,
            use_jax=use_jax,
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

    def state_transition_system(
        self, state: np.ndarray | jnp.ndarray
    ) -> np.ndarray | jnp.ndarray:
        return self.A @ state

    def state_transition_input(
        self, state: np.ndarray | jnp.ndarray
    ) -> np.ndarray | jnp.ndarray:
        return self.B
