import typing

from safe_autonomy_simulation.simulator import Simulator, SimulatorValidator
from safe_autonomy_simulation.docking.point_model import CWHSpacecraft


class DockingSimulatorValidator(SimulatorValidator):
    entities: typing.Dict[str, CWHSpacecraft]

    class Config:
        arbitrary_types_allowed = True


class DockingSimulator(Simulator):
    @property
    def get_sim_validator(self):
        return DockingSimulatorValidator

    def reset(self):
        for _, entity in self.entities.items():
            entity.reset()
        super().reset()

    def step(self):
        step_size = 1 / self.frame_rate
        for _, entity in self.entities.items():
            entity.step(step_size=step_size)  # TODO: add ability to modify control action
        super().step()

    def info(self):
        entity_states = {entity.name: entity.state for _, entity in self.entities.items()}
        return entity_states

    def add_controls(self, control_dict: dict):
        """Add controls to the sim entities control queues. Expects a dict of entity_name: control_to_add items."""
        for e_name, e_control in control_dict.items():
            self.entities[e_name].add_control(e_control)

    @property
    def entities(self):
        """
        Set of simulator entities
        """
        return self.config.entities


if __name__ == "__main__":
    sc1 = CWHSpacecraft(name="sc1")
    sc2 = CWHSpacecraft(name="sc2", x_dot=2)
    entities = {
        "sc1": sc1,
        "sc2": sc2
    }
    sim = DockingSimulator(frame_rate=1, entities=entities)
    sim.reset()

    control_dict = {
        "sc1": [0, 1, 0],
        "sc2": [1, 0, 0]
    }

    step_no = 0
    add_action = 10
    while True:
        if step_no % add_action == 0:
            sim.add_controls(control_dict=control_dict)
        sim.step()
        sc1_control = sim.entities["sc1"].last_control
        sc2_control = sim.entities["sc2"].last_control
        print(f"Step: {step_no}\tSC1 Control: {sc1_control}\tSC2 Control: {sc2_control}")
        step_no += 1
