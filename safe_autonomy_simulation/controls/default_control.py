"""A default control queue with an empty default control vector."""

import numpy as np
import safe_autonomy_simulation.controls.control_queue as c


class DefaultControlQueue(c.ControlQueue):
    """A control queue with an empty default control vector."""

    def __init__(self):
        super().__init__(default_control=np.empty(0))