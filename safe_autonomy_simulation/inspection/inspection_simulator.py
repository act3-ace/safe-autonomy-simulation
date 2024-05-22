"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Safe Autonomy Simulation.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a simulator for a spacecraft inspection environment
using a 6DOF or point mass spacecraft model with or without illumination.
"""

import typing

from safe_autonomy_simulation.inspection.sun import SunEntity
from safe_autonomy_simulation.simulator import Simulator
from safe_autonomy_simulation.inspection.inspector import Inspector, SixDOFInspector
from safe_autonomy_simulation.inspection.target import Target, SixDOFTarget


class InspectionSimulator(Simulator):
    """
    Inspection simulator for a spacecraft inspection environment.

    If a sun entity is provided, the simulator assumes illumination of the inspection targets.
    Illumination is calculated using binary ray tracing if binary_ray is True. Otherwise, the
    simulator assumes full ray tracing.

    Parameters
    ----------
    frame_rate : float
        simulation frame rate
    inspectors : Union[List[Inspector], List[SixDOFInspector]]
        list of inspectors
    targets : Union[List[Target], List[SixDOFTarget]]
        list of inspection targets
    sun : Union[SunEntity, None], optional
        sun entity, by default None
    binary_ray : bool, optional
        binary ray tracing, by default False
    """

    def __init__(
        self,
        frame_rate: float,
        inspectors: typing.Union[typing.List[Inspector], typing.List[SixDOFInspector]],
        targets: typing.Union[typing.List[Target], typing.List[SixDOFTarget]],
        sun: typing.Union[SunEntity, None] = None,
        binary_ray: bool = False,
    ):
        self._inspectors = inspectors
        self._targets = targets
        self._sun = sun
        self._binary_ray = binary_ray

        entities = inspectors + targets
        if sun is not None:
            entities.append(sun)
        super().__init__(frame_rate=frame_rate, entities={e.name: e for e in entities})

    def reset(self):
        super().reset()
        # update inspection points statuses after all entities have been reset
        self._update_inspected()

    def step(self):
        super().step()
        # update inspection points statuses after all entities have been stepped
        self._update_inspected()

    def _update_inspected(self):
        """Update inspection point statuses for all targets"""
        for inspector in self.inspectors:
            for target in self.targets:
                target.inspection_points.update_points_inspection_status(
                    camera=inspector.camera, sun=self.sun, binary_ray=self.binary_ray
                )

    @property
    def inspectors(
        self,
    ) -> typing.Union[typing.List[Inspector], typing.List[SixDOFInspector]]:
        """List of inspectors

        Returns
        -------
        typing.List[CWHSpacecraft]
            list of inspectors
        """
        return self._inspectors

    @property
    def targets(self) -> typing.Union[typing.List[Target], typing.List[SixDOFTarget]]:
        """List of inspection targets

        Returns
        -------
        typing.List[CWHSpacecraft]
            list of targets
        """
        return self._targets

    @property
    def sun(self) -> typing.Union[SunEntity, None]:
        """Sun entity

        Returns
        -------
        typing.Union[SunEntity, None]
            sun entity or None if no sun entity is present
        """
        return self._sun

    @property
    def binary_ray(self) -> bool:
        """Binary ray tracing

        Returns
        -------
        bool
            True if binary ray tracing is enabled, False otherwise
        """
        return self._binary_ray
