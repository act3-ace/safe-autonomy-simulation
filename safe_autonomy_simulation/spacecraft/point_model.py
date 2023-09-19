"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a 3D point mass spacecraft with Clohessy-Wilshire physics dynamics in non-inertial orbital
Hill's reference frame.
"""

from typing import Union

import numpy as np
import pint
from pydantic import validator
from scipy.spatial.transform import Rotation

from safe_autonomy_simulation.base_models import (
    BaseEntity,
    BaseEntityValidator,
    BaseLinearODESolverDynamics,
    BaseUnits,
    build_unit_conversion_validator_fn,
)
from safe_autonomy_simulation.spacecraft.utils import generate_cwh_matrices

M_DEFAULT = 12
N_DEFAULT = 0.001027


class CWHSpacecraftValidator(BaseEntityValidator):
    """
    Validator for CWHSpacecraft kwargs.

    Parameters
    ----------
    x: float or pint.Quantity
        Length 1, x position value
    y: float or pint.Quantity
        Length 1, y position value
    z: float or pint.Quantity
        Length 1, z position value
    x_dot: float or pint.Quantity
        Length 1, x velocity value
    y_dot: float or pint.Quantity
        Length 1, y velocity value
    z_dot: float or pint.Quantity
        Length 1, z velocity value

    Raises
    ------
    ValueError
        Improper list lengths for parameters 'x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot'
    """

    x: Union[float, pint.Quantity] = 0
    y: Union[float, pint.Quantity] = 0
    z: Union[float, pint.Quantity] = 0
    x_dot: Union[float, pint.Quantity] = 0
    y_dot: Union[float, pint.Quantity] = 0
    z_dot: Union[float, pint.Quantity] = 0

    # validators
    _unit_validator_position = validator("x", "y", "z", allow_reuse=True)(build_unit_conversion_validator_fn("meters"))
    _unit_validator_velocity = validator("x_dot", "y_dot", "z_dot", allow_reuse=True)(build_unit_conversion_validator_fn("meters/second"))


class CWHSpacecraft(BaseEntity):
    """
    3D point mass spacecraft with +/- xyz thrusters and Clohessy-Wiltshire dynamics in Hill's reference frame.

    States
        x
        y
        z
        x_dot
        y_dot
        z_dot

    Controls
        thrust_x
            range = [-1, 1] Newtons
        thrust_y
            range = [-1, 1] Newtons
        thrust_z
            range = [-1, 1] Newtons

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12.
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    trajectory_samples : int
        number of trajectory samples the generate and store on steps
    integration_method: str
        Numerical integration method passed to dynamics model. See BaseODESolverDynamics.
    kwargs:
        Additional keyword arguments passed to CWHSpacecraftValidator.
    """

    base_units = BaseUnits("meters", "seconds", "radians")

    def __init__(self, m=M_DEFAULT, n=N_DEFAULT, trajectory_samples=0, integration_method="RK45", **kwargs):
        dynamics = CWHDynamics(
            m=m,
            n=n,
            trajectory_samples=trajectory_samples,
            integration_method=integration_method,
        )
        self._state = np.array([])

        control_map = {
            "thrust_x": 0,
            "thrust_y": 1,
            "thrust_z": 2,
        }

        super().__init__(dynamics, control_default=np.zeros((3, )), control_min=-1, control_max=1, control_map=control_map, **kwargs)

    @classmethod
    def _get_config_validator(cls):
        return CWHSpacecraftValidator

    def __eq__(self, other):
        if isinstance(other, CWHSpacecraft):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = (eq and (self.orientation.as_euler("ZYX") == other.orientation.as_euler("ZYX")).all())
            return eq
        return False

    def _build_state(self):
        state = np.array(
            [self.config.x, self.config.y, self.config.z] + [self.config.x_dot, self.config.y_dot, self.config.z_dot],
            dtype=np.float32,
        )

        return state

    @property
    def x(self):
        """get x"""
        return self._state[0]

    @property
    def y(self):
        """get y"""
        return self._state[1]

    @property
    def z(self):
        """get z"""
        return self._state[2]

    @property
    def x_dot(self):
        """get x_dot, the velocity component in the x direction"""
        return self._state[3]

    @property
    def x_dot_with_units(self):
        """Get x_dot as a pint.Quantity with units"""
        return self.ureg.Quantity(self.x_dot, self.base_units.velocity)

    @property
    def y_dot(self):
        """get y_dot, the velocity component in the y direction"""
        return self._state[4]

    @property
    def y_dot_with_units(self):
        """Get y_dot as a pint.Quantity with units"""
        return self.ureg.Quantity(self.y_dot, self.base_units.velocity)

    @property
    def z_dot(self):
        """get z_dot, the velocity component in the z direction"""
        return self._state[5]

    @property
    def z_dot_with_units(self):
        """Get z_dot as a pint.Quantity with units"""
        return self.ureg.Quantity(self.z_dot, self.base_units.velocity)

    @property
    def position(self):
        """get 3d position vector"""
        return self._state[0:3].copy()

    @property
    def orientation(self):
        """
        Get orientation of CWHSpacecraft, which is always an identity rotation as a point mass model doesn't rotate.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Rotation transformation of the entity's local reference frame basis vectors in the global reference frame.
            i.e. applying this rotation to [1, 0, 0] yields the entity's local x-axis in the global frame.
        """
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.from_quat([0, 0, 0, 1])

    @property
    def velocity(self):
        """Get 3d velocity vector"""
        return self._state[3:6].copy()


class CWHDynamics(BaseLinearODESolverDynamics):
    """
    State transition implementation of 3D Clohessy-Wiltshire dynamics model.

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12
    n: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027
    kwargs:
        Additional keyword arguments passed to parent class BaseLinearODESolverDynamics constructor
    """

    def __init__(self, m=M_DEFAULT, n=N_DEFAULT, **kwargs):
        self.m = m  # kg
        self.n = n  # rads/s

        A, B = generate_cwh_matrices(self.m, self.n, "3d")

        super().__init__(A=A, B=B, **kwargs)
