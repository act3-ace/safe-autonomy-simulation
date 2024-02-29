"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a simulator for a spacecraft inspection environment
using a 6DOF or point mass spacecraft model.
"""

import typing

import numpy as np

from safe_autonomy_simulation.inspection.illumination import IlluminationParams
from safe_autonomy_simulation.inspection.inspection_points import InspectionPoints, InspectionPointsValidator
from safe_autonomy_simulation.inspection.sun_model import SunEntity
from safe_autonomy_simulation.simulator import ControlledDiscreteSimulator, DiscreteSimulatorValidator
from safe_autonomy_simulation.spacecraft.point_model import CWHSpacecraft
from safe_autonomy_simulation.spacecraft.sixdof_model import SixDOFSpacecraft


class InspectionSimulatorValidator(DiscreteSimulatorValidator):
    """
    A validator for the InspectionSimulator config.

    entities: typing.Dict[str, typing.Union[SixDOFSpacecraft, CWHSpacecraft]]
        dict of {entity_name: simulation_entity}
    inspectors: typing.List[str]
        list of entities performing inspection
    illumination_params: typing.Union[IlluminationValidator, None] = None
        dict of illumination parameters
    sensor_fov: float
        field of view of the sensor (radians). By default pi (180 degrees)
    initial_sensor_unit_vec: list
        If using the 6DOF spacecraft model, initial unit vector along sensor boresight.
        By default [1., 0., 0.]
    inspection_points_map: dict
        A map of entity name strings to InspectionPoints objects, which track the inspection progress of each entity.
    """

    entities: typing.Dict[str, typing.Union[SixDOFSpacecraft, CWHSpacecraft, SunEntity]]
    inspectors: typing.List[str]
    illumination_params: typing.Union[IlluminationParams, None] = None
    sensor_fov: float = np.pi
    initial_sensor_unit_vec: list = [1.0, 0.0, 0.0]
    inspection_points_map: typing.Dict[str, InspectionPointsValidator]

    class Config:
        """Allow arbitrary types for Parameter"""

        arbitrary_types_allowed = True


class InspectionSimulator(ControlledDiscreteSimulator):
    """
    Simulator for CWH Inspection Task. Interfaces CWH platforms with underlying CWH entities in inspection simulation.
    """

    @property
    def get_sim_validator(self):
        return InspectionSimulatorValidator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inspection_points_map = {}
        self.sun_angle = 0.0
        self.priority_vector = np.zeros(3)
        self.inspectors = {inspector_name: self.entities[inspector_name] for inspector_name in self.config.inspectors}

    def create_inspection_points_map(self):
        """
        create map of inspection points for each entity
        """
        points_map = {}
        for (
            entity_name,
            inspection_points_validator,
        ) in self.config.inspection_points_map.items():
            # TODO: there must be a better way to use validators
            parent_entity = self.entities[entity_name]
            points_map[entity_name] = InspectionPoints(
                parent_entity=parent_entity,
                radius=inspection_points_validator.radius,
                num_points=inspection_points_validator.num_points,
                points_algorithm=inspection_points_validator.points_algorithm,
                sensor_fov=self.config.sensor_fov,
                initial_sensor_unit_vec=self.config.initial_sensor_unit_vec,
                illumination_params=self.config.illumination_params,
                priority_vector=self.priority_vector,
            )

        return points_map

    def reset(self, priority_vec_azimuth=0.0, priority_vec_elevation=0.0):
        super().reset()
        if self.config.illumination_params is not None:
            self.sun_angle = self.entities["sun"].theta

        self._get_initial_priority_vector(priority_vec_azimuth, priority_vec_elevation)

        # reset points map
        self.inspection_points_map = self.create_inspection_points_map()
        for points in self.inspection_points_map.values():
            points.update_points_position()

        # illuminate
        if self.config.illumination_params:
            # pass sun_angle to InspectionPoints objs
            for points in self.inspection_points_map.values():
                points.set_sun_angle(self.sun_angle)

        self._update_inspection_points_statuses()

    def _get_initial_priority_vector(self, priority_vec_azimuth, priority_vec_elevation):
        """Get the initial priority vector for weighting points"""
        self.priority_vector[0] = np.cos(priority_vec_azimuth) * np.cos(priority_vec_elevation)
        self.priority_vector[1] = np.sin(priority_vec_azimuth) * np.cos(priority_vec_elevation)
        self.priority_vector[2] = np.sin(priority_vec_elevation)

    def step(self):
        super().step()

        # update inspection points positions
        for points in self.inspection_points_map.values():
            points.update_points_position()

        # illuminate
        if self.config.illumination_params:
            self.sun_angle = self.entities["sun"].theta
            # pass sun_angle to InspectionPoints objs
            for points in self.inspection_points_map.values():
                points.set_sun_angle(self.sun_angle)

        # update inspection points statuses
        self._update_inspection_points_statuses()

    def _update_inspection_points_statuses(self):
        for inspection_entity_name, points in self.inspection_points_map.items():
            for inspector_entity_name, inspector_entity in self.inspectors.items():
                if inspection_entity_name is not inspector_entity_name:
                    points.update_points_inspection_status(inspector_entity)
