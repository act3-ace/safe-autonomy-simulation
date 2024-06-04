"""Pass-through dynamics model"""

import typing
import numpy as np
import safe_autonomy_simulation.dynamics.dynamics as d


class PassThroughDynamics(d.Dynamics):
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
        state_min: typing.Union[float, np.ndarray] = -np.inf,
        state_max: typing.Union[float, np.ndarray] = np.inf,
        angle_wrap_centers: typing.Union[np.ndarray, None] = None,
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
