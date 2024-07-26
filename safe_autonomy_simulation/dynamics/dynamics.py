"""Base class for state transition dynamics models of entities in the simulation environment."""

import typing

try:
    import jax.numpy as np
except ImportError:
    import numpy as np


class Dynamics:
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
    """

    def __init__(
        self,
        state_min: typing.Union[float, np.ndarray] = -np.inf,
        state_max: typing.Union[float, np.ndarray] = np.inf,
    ):
        self.np = np
        self.state_min = state_min
        self.state_max = state_max

    def step(
        self, step_size: float, state: np.ndarray, control: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
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
        return next_state, state_dot

    def _step(
        self, step_size: float, state: np.ndarray, control: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
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
