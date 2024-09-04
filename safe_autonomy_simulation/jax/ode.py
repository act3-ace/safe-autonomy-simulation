import jax
import typing
import jax.numpy as jnp
import scipy.integrate as integrate


@jax.jit
def clip_state_dot(
    state_dot: jnp.ndarray, s_min: jnp.ndarray | float, s_max: jnp.ndarray | float
) -> jnp.ndarray:
    """Clip state derivatives to be within the given bounds.

    Parameters
    ----------
    state_dot : jnp.ndarray
        State derivatives.
    s_min : jnp.ndarray | float
        Minimum state derivative values.
    s_max : jnp.ndarray | float
        Maximum state derivative values.

    Returns
    -------
    jnp.ndarray
        Clipped state derivatives.
    """
    return jnp.clip(state_dot, s_min, s_max)


@jax.jit
def clip_state_dot_at_state_limits(
    state: jnp.ndarray,
    state_dot: jnp.ndarray,
    s_min: jnp.ndarray | float,
    s_max: jnp.ndarray | float,
) -> jnp.ndarray:
    """Clip state derivatives such that the state remains within the given bounds.

    Parameters
    ----------
    state : jnp.ndarray
        State values.
    state_dot : jnp.ndarray
        State derivatives.
    s_min : jnp.ndarray | float
        Minimum state values.
    s_max : jnp.ndarray | float
        Maximum state values.

    Returns
    -------
    jnp.ndarray
        Clipped state derivatives.
    """
    lower_bounded_states = state <= s_min
    upper_bounded_states = state >= s_max

    lower_bounded_clipped = jnp.clip(state_dot, 0, jnp.inf)
    upper_bounded_clipped = jnp.clip(state_dot, jnp.inf, 0)

    state_dot = jnp.where(lower_bounded_states, lower_bounded_clipped, state_dot)
    state_dot = jnp.where(upper_bounded_states, upper_bounded_clipped, state_dot)

    return state_dot


@jax.jit(static_argnames=("trajectory_samples",))
def step_rk45(
    f: typing.Callable,
    step_size: float,
    state: jnp.ndarray,
    control: jnp.ndarray,
    trajectory_samples: int = 0,
):
    """Perform a single Runge-Kutta 4th order step.

    Parameters
    ----------
    f : Callable
        Function to compute state derivatives
    step_size : float
        Step size.
    state : jnp.ndarray
        State values.
    control : jnp.ndarray
        Control values.
    trajectory_samples : int, optional
        Number of samples to take along the trajectory, by default 0.


    Returns
    -------
    jnp.ndarray
        Next state values.
    jnp.ndarray
        Next state derivatives.
    jnp.ndarray
        Trajectory values.
    jnp.ndarray
        Trajectory time values.
    """
    t_eval = None
    if trajectory_samples > 0:
        t_eval = jnp.linspace(0, step_size, trajectory_samples + 1)[1:]

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


@jax.jit
def step_euler(
    f: typing.Callable,
    step_size: float,
    state: jnp.ndarray,
    control: jnp.ndarray,
):
    """Perform a single Euler step.

    Parameters
    ----------
    f : Callable
        Function to compute state derivatives
    step_size : float
        Step size.
    state : jnp.ndarray
        State values.
    control : jnp.ndarray
        Control values.

    Returns
    -------
    jnp.ndarray
        Next state values.
    jnp.ndarray
        Next state derivatives.
    jnp.ndarray
        Trajectory values.
    jnp.ndarray
        Trajectory time values.
    """
    state_dot = f(step_size, state, control)
    next_state = state + step_size * state_dot

    trajectory = jnp.vstack([state, next_state])
    trajectory_t = jnp.array([0, step_size])

    return next_state, state_dot, trajectory, trajectory_t


@jax.jit
def affine_transition(
    state: jnp.ndarray,
    control: jnp.ndarray,
    f: typing.Callable,
    g: typing.Callable,
):
    """Affine transition function.

    Computes state derivatives using the affine transition function:
    x_dot = f(x) + g(x)u

    Parameters
    ----------
    state : jnp.ndarray
        State values.
    control : jnp.ndarray
        Control values.
    f : Callable
        State transition function.
    g : Callable
        Control transition function.

    Returns
    -------
    jnp.ndarray
        State derivatives.
    """
    return f(state) + g(state) @ control
