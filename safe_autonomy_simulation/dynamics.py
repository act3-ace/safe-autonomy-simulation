"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module provides base implementations for entity dynamics
"""

from __future__ import annotations

import abc
from types import ModuleType
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import scipy.integrate

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
else:
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental.ode import odeint
    except ImportError:
        jax = None
        jnp = None
        odeint = None


class Dynamics(abc.ABC):
    """
    State transition implementation for a physics dynamics model. Used by entities to compute their next state when
    their step() method is called.

    Parameters
    ----------
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
    angle_wrap_centers: np.ndarray, optional
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within
        [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
        By default, None.
    use_jax : bool, optional
        True if using jax version of numpy/scipy. By default, False
    """

    def __init__(
        self,
        state_min: Union[float, np.ndarray] = -np.inf,
        state_max: Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: Union[np.ndarray, None] = None,
        use_jax: bool = False,
    ):
        self.state_min = state_min
        self.state_max = state_max
        self.angle_wrap_centers = angle_wrap_centers
        self.use_jax = use_jax

        self.np: ModuleType
        if use_jax:
            if jax is None:  # pylint: disable=used-before-assignment
                raise ImportError(
                    "Failed to import jax. Make sure to install jax if using the `use_jax` option"
                )
            self.np = jnp
        else:
            self.np = np

    def step(
        self, step_size: float, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the dynamics state transition from the current state and control input.

        Parameters
        ----------
        step_size : float
            Duration of the simulation step in seconds.
        state : np.ndarray
            Current state of the system at the beginning of the simulation step.
        control : np.ndarray
            Control vector of the dynamics model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of the system's next state and the state's instantaneous time derivative at the end of the step
        """
        next_state, state_dot = self._step(step_size, state, control)
        next_state = self.np.clip(next_state, self.state_min, self.state_max)
        next_state = self._wrap_angles(next_state)
        return next_state, state_dot

    def _wrap_angles(self, state: Union[np.ndarray, jnp.ndarray]):
        """Wraps angles in state to be within [0, 2 * pi] range
        
        Parameters
        ----------
        state : Union[np.ndarray, jnp.ndarray]
            State vector of the system.
            
        Returns
        -------
        Union[np.ndarray, jnp.ndarray]
            State vector with angles wrapped to be within [0, 2 * pi] range
        """
        if self.angle_wrap_centers is not None:
            needs_wrap = self.np.logical_not(self.np.isnan(self.angle_wrap_centers))

            wrapped_state = (
                ((state + np.pi) % (2 * np.pi)) - np.pi + self.angle_wrap_centers
            )

            output_state = self.np.where(needs_wrap, wrapped_state, state)
        else:
            output_state = state

        return output_state

    @abc.abstractmethod
    def _step(
        self, step_size: float, state: np.ndarray, control: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the next state and state derivative of the system
        
        This method should be implemented by the subclass to compute the next state and state derivative of the system

        Parameters
        ----------
        step_size : float
            Duration of the simulation step in seconds.
        state : np.ndarray
            Current state of the system at the beginning of the simulation step.
        control : np.ndarray
            Control vector of the dynamics model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of the system's next state and the state's instantaneous time derivative at the end of the step
        """
        raise NotImplementedError


class PassThroughDynamics(Dynamics):
    """
    State transition implementation for a pass-through dynamics model. The next state is equal to the current state
    and the state derivative is equal to the control input.

    Parameters
    ----------
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
    angle_wrap_centers: np.ndarray, optional
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within
        [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
        By default, None.
    use_jax : bool, optional
        True if using jax version of numpy/scipy. By default, False
    """

    def __init__(
        self,
        state_min: Union[float, np.ndarray] = -np.inf,
        state_max: Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: Union[np.ndarray, None] = None,
        use_jax: bool = False,
    ):
        super().__init__(
            state_min=state_min,
            state_max=state_max,
            angle_wrap_centers=angle_wrap_centers,
            use_jax=use_jax,
        )

    def _step(self, step_size: float, state: np.ndarray, control: np.ndarray):
        return state, control


class ODESolverDynamics(Dynamics):
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
    angle_wrap_centers: np.ndarray, optional
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within
        [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
        By default, None.
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'RK45_JAX', 'Euler'].
        'RK45' is slow but very accurate.
        'RK45_JAX' is very accurate, and fast when JIT compiled but otherwise very slow. 'use_jax' must be set to True.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    use_jax : bool, optional
        True if using jax version of numpy/scipy. By default, False
    """

    def __init__(
        self,
        trajectory_samples: int = 0,
        state_min: Union[float, np.ndarray] = -np.inf,
        state_max: Union[float, np.ndarray] = np.inf,
        state_dot_min: Union[float, np.ndarray] = -np.inf,
        state_dot_max: Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: Union[np.ndarray, None] = None,
        integration_method: str = "RK45",
        use_jax: bool = False,
    ):
        super().__init__(
            state_min=state_min,
            state_max=state_max,
            angle_wrap_centers=angle_wrap_centers,
            use_jax=use_jax,
        )

        self.integration_method = integration_method
        self.state_dot_min = state_dot_min
        self.state_dot_max = state_dot_max

        assert isinstance(
            trajectory_samples, int
        ), "trajectory_samples must be an integer"
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

    @abc.abstractmethod
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
        elif self.integration_method == "RK45_JAX":
            if not self.use_jax:
                raise ValueError("use_jax must be set to True if using RK45_JAX")

            assert (
                self.trajectory_samples <= 0
            ), "trajectory sampling not currently supported with rk45 jax integration"

            sol = odeint(  # pylint: disable=used-before-assignment
                self.compute_state_dot_jax,
                state,
                jnp.linspace(0.0, step_size, 11),
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
        return self._compute_state_dot(t, state, control)


class ControlAffineODESolverDynamics(ODESolverDynamics):
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
    angle_wrap_centers: np.ndarray, optional
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within
        [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
        By default, None.
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'RK45_JAX', 'Euler'].
        'RK45' is slow but very accurate.
        'RK45_JAX' is very accurate, and fast when JIT compiled but otherwise very slow. 'use_jax' must be set to True.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    use_jax : bool, optional
        True if using jax version of numpy/scipy. By default, False
    """

    def _compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray):
        state_dot = (
            self.state_transition_system(state)
            + self.state_transition_input(state) @ control
        )
        return state_dot

    @abc.abstractmethod
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

    @abc.abstractmethod
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


class LinearODESolverDynamics(ControlAffineODESolverDynamics):
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
    angle_wrap_centers: np.ndarray, optional
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within
        [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
        By default, None.
    integration_method : string, optional
        Numerical integration method used by dynamics solver. One of ['RK45', 'RK45_JAX', 'Euler'].
        'RK45' is slow but very accurate.
        'RK45_JAX' is very accurate, and fast when JIT compiled but otherwise very slow. 'use_jax' must be set to True.
        'Euler' is fast but very inaccurate.
        By default, 'RK45'.
    use_jax : bool, optional
        True if using jax version of numpy/scipy. By default, False
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        trajectory_samples: int = 0,
        state_min: Union[float, np.ndarray] = -np.inf,
        state_max: Union[float, np.ndarray] = np.inf,
        state_dot_min: Union[float, np.ndarray] = -np.inf,
        state_dot_max: Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: Union[np.ndarray, None] = None,
        integration_method: str = "RK45",
        use_jax: bool = False,
    ):
        super().__init__(
            trajectory_samples=trajectory_samples,
            state_min=state_min,
            state_max=state_max,
            state_dot_min=state_dot_min,
            state_dot_max=state_dot_max,
            angle_wrap_centers=angle_wrap_centers,
            integration_method=integration_method,
            use_jax=use_jax,
        )

        assert (
            len(A.shape) == 2
        ), f"A must be square matrix. Instead got shape {A.shape}"
        assert (
            len(B.shape) == 2
        ), f"A must be square matrix. Instead got shape {B.shape}"
        assert (
            A.shape[0] == A.shape[1]
        ), f"A must be a square matrix, not dimension {A.shape}"
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
