"""This package implements various control queues.

Control queues are used to store control vectors that are applied to entities in the simulation.

All control queues are instances of the ControlQueue class, which is an abstract base class that
defines the interface for all control queues in the simulation.
"""

from safe_autonomy_simulation.controls.control_queue import ControlQueue
from safe_autonomy_simulation.controls.no_control import NoControl

__all__ = [
    "ControlQueue",  # base control queue
    "NoControl",  # control queue with an empty default control vector
]
