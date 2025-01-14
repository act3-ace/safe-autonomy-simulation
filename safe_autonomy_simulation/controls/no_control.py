"""A default control queue with an empty default control vector."""

import numpy as np
import safe_autonomy_simulation.controls.control_queue as c


class NoControl(c.ControlQueue):
    """A control queue with an empty default control vector.

    This control queue will always return an empty control vector.
    """

    def __init__(self):
        super().__init__(
            default_control=np.empty(0),
            control_min=np.empty(0),
            control_max=np.empty(0),
        )

    def add_control(self, control: np.ndarray | list):
        pass
