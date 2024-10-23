import typing
import numpy as np
import scipy.integrate as integrate


def clip_state_dot(
    state_dot: np.ndarray, s_min: np.ndarray | float, s_max: np.ndarray | float
) -> np.ndarray:
    """Clip state derivatives to be within the given bounds.

    Parameters
    ----------
    state_dot : np.ndarray
        State derivatives.
    s_min : np.ndarray | float
        Minimum state derivative values.
    s_max : np.ndarray | float
        Maximum state derivative values.

    Returns
    -------
    np.ndarray
        Clipped state derivatives.
    """
    return np.clip(state_dot, s_min, s_max)


def clip_state_dot_at_state_limits(
    state: np.ndarray,
    state_dot: np.ndarray,
    s_min: np.ndarray | float,
    s_max: np.ndarray | float,
) -> np.ndarray:
    """Clip state derivatives such that the state remains within the given bounds.

    Parameters
    ----------
    state : np.ndarray
        State values.
    state_dot : np.ndarray
        State derivatives.
    s_min : np.ndarray | float
        Minimum state values.
    s_max : np.ndarray | float
        Maximum state values.

    Returns
    -------
    np.ndarray
        Clipped state derivatives.
    """
    lower_bounded_states = state <= s_min
    upper_bounded_states = state >= s_max

    lower_bounded_clipped = np.clip(state_dot, 0, np.inf)
    upper_bounded_clipped = np.clip(state_dot, -np.inf, 0)

    state_dot = np.where(lower_bounded_states, lower_bounded_clipped, state_dot)
    state_dot = np.where(upper_bounded_states, upper_bounded_clipped, state_dot)

    return state_dot


def step_rk45(
    f: typing.Callable,
    step_size: float,
    state: np.ndarray,
    control: np.ndarray,
    trajectory_samples: int = 0,
):
    """Perform a single Runge-Kutta 4th order step.

    Parameters
    ----------
    f : Callable
        Function to compute state derivatives
    step_size : float
        Step size.
    state : np.ndarray
        State values.
    control : np.ndarray
        Control values.
    trajectory_samples : int, optional
        Number of samples to take along the trajectory, by default 0.


    Returns
    -------
    np.ndarray
        Next state values.
    np.ndarray
        Next state derivatives.
    np.ndarray
        Trajectory values.
    np.ndarray
        Trajectory time values.
    """
    t_eval = None
    if trajectory_samples > 0:
        t_eval = np.linspace(0, step_size, trajectory_samples + 1)[1:]

    sol = integrate.solve_ivp(
        f,
        (0, step_size),
        state,
        args=(control,),
        t_eval=t_eval,
    )

    trajectory = sol.y.T
    trajectory_t = sol.t

    next_state = sol.y[:, -1]  # save last timestep of integration solution
    state_dot = f(step_size, next_state, control)

    return next_state, state_dot, trajectory, trajectory_t


def step_euler(
    f: typing.Callable,
    step_size: float,
    state: np.ndarray,
    control: np.ndarray,
):
    """Perform a single Euler step.

    Parameters
    ----------
    f : Callable
        Function to compute state derivatives
    step_size : float
        Step size.
    state : np.ndarray
        State values.
    control : np.ndarray
        Control values.

    Returns
    -------
    np.ndarray
        Next state values.
    np.ndarray
        Next state derivatives.
    np.ndarray
        Trajectory values.
    np.ndarray
        Trajectory time values.
    """
    state_dot = f(step_size, state, control)
    next_state = state + step_size * state_dot

    trajectory = np.vstack([state, next_state])
    trajectory_t = np.array([0, step_size])

    return next_state, state_dot, trajectory, trajectory_t


def affine_transition(
    state: np.ndarray,
    control: np.ndarray,
    f: typing.Callable,
    g: typing.Callable,
):
    """Affine transition function.

    Computes state derivatives using the affine transition function:
    x_dot = f(x) + g(x)u

    Parameters
    ----------
    state : np.ndarray
        State values.
    control : np.ndarray
        Control values.
    f : Callable
        State transition function.
    g : Callable
        Control transition function.

    Returns
    -------
    np.ndarray
        State derivatives.
    """
    return f(state) + g(state) @ control
