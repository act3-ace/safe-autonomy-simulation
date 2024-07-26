"""Control input queue for entities."""

import typing
import queue
import warnings

try:
    import jax.numpy as np
except ImportError:
    import numpy as np


class ControlQueue:
    """
    A queue of entity controls to be applied to an Entity.

    Parameters
    ----------
    default_control : np.ndarray
        Default control vector used when the control queue is empty. Typically 0 or neutral for each actuator.
    control_min : Union[float, None], optional
        Optional minimum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_max : Union[float, None], optional
        Optional maximum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.

    Attributes
    ----------
    controls : SimpleQueue
        Queue of control vectors to be applied to the entity.
    default_control : np.ndarray
        Default control vector used when the control queue is empty. Typically 0 or neutral for each actuator.
    control_min : Union[float, np.ndarray, None], optional
        Minimum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_max : Union[float, np.ndarray, None], optional
        Maximum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    """

    def __init__(
        self,
        default_control: np.ndarray,
        control_min: typing.Union[float, np.ndarray, None] = None,
        control_max: typing.Union[float, np.ndarray, None] = None,
    ):
        self.controls: queue.SimpleQueue = queue.SimpleQueue()
        self.default_control = default_control
        if control_min is not None and not isinstance(control_min, np.ndarray):
            control_min = np.array(control_min)
        self.control_min = control_min
        if control_max is not None and not isinstance(control_max, np.ndarray):
            control_max = np.array(control_max)
        self.control_max = control_max

    def reset(self):
        """Clears the control queue"""
        while not self.controls.empty():
            self.controls.get()

    def empty(self) -> bool:
        """
        Check if the control queue is empty.

        Returns
        -------
        bool
            True if the control queue is empty, False otherwise.
        """
        return self.controls.empty()

    def next_control(self) -> np.ndarray:
        """Removes and returns the next control in the control queue.

        If control queue is empty, returns the default control.

        Returns
        -------
        np.ndarray
            Next control in the control queue or default control.
        """
        if self.empty():
            control = self.default_control
        else:
            control = self.controls.get()

        return control

    def add_control(self, control: typing.Union[np.ndarray, list]):
        """Adds a control to the end of the control queue.

        Parameters
        ----------
        control : Union[np.ndarray, list]
            Control vector to be added to the control queue.
        """

        if not isinstance(control, np.ndarray):
            control = np.array(control)

        # enforce control bounds (if any)
        if (self.control_min is not None and np.any(control < self.control_min)) or (
            self.control_max is not None and np.any(control > self.control_max)
        ):
            warnings.warn(
                f"Control input exceeded limits. Clipping to range ({self.control_min}, {self.control_max})"
            )
            control = np.clip(control, self.control_min, self.control_max)

        self.controls.put(control)
