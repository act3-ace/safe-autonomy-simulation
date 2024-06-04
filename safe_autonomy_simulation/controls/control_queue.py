"""Control input queue for entities."""

import typing
import queue
import warnings
import numpy as np

if typing.TYPE_CHECKING:
    import jax.numpy as jnp
else:
    try:
        import jax.numpy as jnp
    except ImportError:
        jnp = None


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
    control_map : Union[dict, None], optional
        Optional mapping for actuator names to their indices in the state vector.
        Allows dictionary action inputs in add_control(). By default None.

    Attributes
    ----------
    controls : SimpleQueue
        Queue of control vectors to be applied to the entity.
    default_control : np.ndarray
        Default control vector used when the control queue is empty. Typically 0 or neutral for each actuator.
    control_min : Union[np.ndarray, None], optional
        Minimum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_max : Union[np.ndarray, None], optional
        Maximum allowable control vector values. Control vectors that exceed this limit are clipped. By default None.
    control_map : Union[dict, None], optional
        Mapping for actuator names to their indices in the state vector.
        Required for dictionary action inputs in add_control(). By default None.
    """

    def __init__(
        self,
        default_control: np.ndarray,
        control_min: typing.Union[float, None] = None,
        control_max: typing.Union[float, None] = None,
        control_map: typing.Union[dict, None] = None,
    ):
        self.controls = queue.SimpleQueue()
        self.default_control = default_control
        self.control_min = control_min
        self.control_max = control_max
        self.control_map = control_map

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

    def next_control(self) -> typing.Union[np.ndarray, jnp.ndarray]:
        """Removes and returns the next control in the control queue.

        If control queue is empty, returns the default control.

        Returns
        -------
        Union[np.ndarray, jnp.ndarray]
            Next control in the control queue or default control.
        """
        if self.empty():
            control = self.default_control
        else:
            control = self.controls.get()

        return control

    def _vectorize_control(self, control: dict) -> np.ndarray:
        control_vector = np.zeros(len(self.default_control))
        for k, v in control.items():
            if k not in self.control_map:
                raise KeyError(
                    f"action '{k}' not found in entity's control_map, "
                    f"please use one of: {self.control_map.keys()}"
                )

            control_vector[self.control_map[k]] = v
        if jnp is not None:
            control_vector = jnp.array(control_vector)
        return control_vector

    def add_control(self, control: typing.Union[np.ndarray, dict, list, jnp.ndarray]):
        """Adds a control to the end of the control queue.

        Parameters
        ----------
        control : Union[np.ndarray, dict, list, jnp.ndarray]
            Control vector to be added to the control queue.
        """
        if isinstance(control, dict):
            assert (
                self.control_map is not None
            ), "Cannot use dict-type action without a control_map "
            control = self._vectorize_control(control)
        elif isinstance(control, list):
            control = np.array(control, dtype=np.float32)
        elif isinstance(control, np.ndarray):
            control = control.copy()
        elif jnp is not None and isinstance(  # pylint: disable=used-before-assignment
            control, jnp.ndarray
        ):  # pylint: disable=used-before-assignment
            control = control.copy()
        else:
            raise ValueError(
                "action must be type dict, list, np.ndarray or jnp.ndarray"
            )

        # enforce control bounds (if any)
        if (self.control_min or self.control_max) and (
            np.any(control < self.control_min) or np.any(control > self.control_max)
        ):
            warnings.warn(
                f"Control input exceeded limits. Clipping to range ({self.control_min}, {self.control_max})"
            )
            control = np.clip(control, self.control_min, self.control_max)

        self.controls.put(control)
