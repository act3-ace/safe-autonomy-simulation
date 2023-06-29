import typing

from safe_autonomy_simulation.simulator import Simulator, SimulatorValidator
from safe_autonomy_simulation.docking.point_model import CWHSpacecraft


class DockingSimulatorValidator(SimulatorValidator):
    entities: typing.List[CWHSpacecraft]

    class Config:
        arbitrary_types_allowed = True


class DockingSimulator(Simulator):
    @property
    def get_sim_validator(self):
        return DockingSimulatorValidator

    def reset(self):
        for entity in self.entities:
            entity.reset()
        super().reset()

    def step(self):
        step_size = 1 / self.frame_rate
        for entity in self.entities:
            entity.step(step_size=step_size)  # TODO: add ability to modify control action
        super().step()

    def info(self):
        entity_states = {entity.name: entity.state for entity in self.entities}
        return entity_states

    @property
    def entities(self):
        """
        Set of simulator entities
        """
        return self.config.entities


if __name__ == "__main__":
    sc1 = CWHSpacecraft(name="sc1")
    sc2 = CWHSpacecraft(name="sc2", x_dot=2)
    sim = DockingSimulator(frame_rate=1, entities=[sc1, sc2])
    sim.reset()

    while True:
        sim.step()
        print(sim.info())
