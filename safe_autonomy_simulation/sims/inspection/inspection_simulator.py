"""This module implements a simulator for an inspection environment with or without illumination."""

import typing

import safe_autonomy_simulation.sims.inspection.sun as sun
import safe_autonomy_simulation.simulator as sim
import safe_autonomy_simulation.sims.inspection.inspector as i
import safe_autonomy_simulation.sims.inspection.target as t


class InspectionSimulator(sim.Simulator):
    """
    Simulator for an inspection environment made up of inspectors and inspection targets.

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
    sun : Union[Sun, None], optional
        sun entity, by default None
    binary_ray : bool, optional
        binary ray tracing, by default False
    """

    def __init__(
        self,
        frame_rate: float,
        inspectors: typing.Union[
            typing.List[i.Inspector], typing.List[i.SixDOFInspector]
        ],
        targets: typing.Union[typing.List[t.Target], typing.List[t.SixDOFTarget]],
        sun: typing.Union[sun.Sun, None] = None,
        binary_ray: bool = True,
    ):
        self._inspectors = inspectors
        self._targets = targets
        self._sun = sun
        self._binary_ray = binary_ray

        entities = inspectors + targets
        if sun is not None:
            entities.append(sun)
        super().__init__(frame_rate=frame_rate, entities=entities)

    def reset(self):
        super().reset()
        # update inspection points statuses after all entities have been reset
        self._update_inspected()

    def update(self):
        # set cameras to point at target if inspectors are 3dof and there is only one target
        # if isinstance(self.inspectors[0], i.Inspector) and len(self.targets) == 1:
        #     target = self.targets[0]
        #     for inspector in self.inspectors:
        #         inspector.camera.point_at(target)

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
    ) -> typing.Union[typing.List[i.Inspector], typing.List[i.SixDOFInspector]]:
        """List of inspectors

        Returns
        -------
        typing.List[CWHSpacecraft]
            list of inspectors
        """
        return self._inspectors

    @property
    def targets(
        self,
    ) -> typing.Union[typing.List[t.Target], typing.List[t.SixDOFTarget]]:
        """List of inspection targets

        Returns
        -------
        typing.List[CWHSpacecraft]
            list of targets
        """
        return self._targets

    @property
    def sun(self) -> typing.Union[sun.Sun, None]:
        """Sun entity

        Returns
        -------
        typing.Union[Sun, None]
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
