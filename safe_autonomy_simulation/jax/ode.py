import jax
import typing
import jax.numpy as jnp
import functools
import diffrax


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


@functools.partial(
    jax.jit,
    static_argnames=(
        "f",
        "step_size",
        "trajectory_samples",
    ),
)
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
    saveat = diffrax.SaveAt(t1=True)
    if trajectory_samples > 0:
        saveat = diffrax.SaveAt(ts=jnp.linspace(0, step_size, trajectory_samples + 1)[1:])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        solver=diffrax.Dopri5(),
        saveat=saveat,
        t0=0,
        t1=step_size,
        dt0=None,
        y0=state,
        args=control,
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
    )

    trajectory = sol.ys
    trajectory_t = sol.ts

    next_state = sol.ys[-1, :]  # save last timestep of integration solution
    state_dot = f(step_size, next_state, control)

    return next_state, state_dot, trajectory, trajectory_t


@functools.partial(jax.jit, static_argnames=("f",))
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


@functools.partial(jax.jit, static_argnames=("f", "g"))
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
    return f(jnp.array(state)) + g(jnp.array(state)) @ jnp.array(control)
