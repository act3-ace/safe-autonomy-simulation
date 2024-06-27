"""
This module implements a sun model in non-inertial orbital Hill's reference frame.
"""

import typing
import numpy as np
import pint

import safe_autonomy_simulation.entities as e
import safe_autonomy_simulation.dynamics as d
import safe_autonomy_simulation.materials as mat
import safe_autonomy_simulation.sims.spacecraft.defaults as defaults


class Sun(e.Point):
    """
    Sun in Hill's reference frame.
    Assumed to rotate in x-y plane with angular velocity "n"

    Parameters
    ----------
    name: str, optional
        name of the entity, by default "sun"
    theta: Union[float, pint.Quantity], optional
        initial angle of the sun in radians, by default 0
    n: float, optional
        mean motion of the sun in rad/s, by default 0.001027
    integration_method: str, optional
        integration method for dynamics, by default "RK45"
    use_jax: bool, optional
        whether to use jax for dynamics, by default False
    material: Material, optional
        material properties of the sun, by default LIGHT
    """

    def __init__(
        self,
        name: str = "sun",
        theta: typing.Union[float, pint.Quantity] = 0,
        n: float = defaults.N_DEFAULT,
        integration_method: str = "RK45",
        use_jax: bool = False,
        material: mat.Material = mat.LIGHT,
    ):
        self._initial_theta = theta
        self._n = n  # rads/s
        dynamics = SunDynamics(
            n=n, integration_method=integration_method, use_jax=use_jax
        )
        super().__init__(
            name=name, dynamics=dynamics, position=np.zeros(3), material=material
        )

    def __eq__(self, other):
        if isinstance(other, Sun):
            return True
        return False

    def build_initial_state(self) -> np.ndarray:
        state = super().build_initial_state()
        state = np.concatenate([state, np.array([self._initial_theta])])
        return state

    @property
    def state(self) -> np.ndarray:
        """Sun entity state vector

        State vector is [theta]

        Returns
        -------
        np.ndarray
            state vector of form [theta]
        """
        return np.array(self._state[-1])

    @property
    def theta(self) -> float:
        """Sun rotation angle

        Returns
        -------
        float
            Sun rotation angle in radians
        """
        return float(self.state)

    @property
    def n(self) -> float:
        """Sun mean motion

        Returns
        -------
        float
            Sun mean motion in rad/s
        """
        return self._n


class SunDynamics(d.ODEDynamics):
    """Dynamics for the sun. Assumed to rotate in x-y plane

    Parameters
    ----------
    n: float, optional
        mean motion of the sun in rad/s, by default 0.001027
    integration_method: str, optional
        integration method for dynamics, by default "RK45"
    use_jax: bool, optional
        whether to use jax for dynamics, by default False
    """

    def __init__(self, n=defaults.N_DEFAULT, integration_method="RK45", use_jax=False):
        self.n = n  # rads/s
        super().__init__(integration_method=integration_method, use_jax=use_jax)

    def _compute_state_dot(
        self, t: float, state: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        return np.array([self.n])
