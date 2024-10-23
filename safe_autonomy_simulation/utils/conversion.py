import numpy as np
import safe_autonomy_simulation

if safe_autonomy_simulation.jax_available():
    import jax.numpy as jnp


def cast_jax(arr: np.ndarray, use_jax: bool) -> np.ndarray:
    """Cast numpy array to jax array if jax is available and use_jax is True.

    This function returns the original array otherwise.

    Parameters
    ----------
    arr : np.ndarray
        Array to cast to jax array
    use_jax : bool
        Cast to jax array if True
    
    Returns
    -------
    np.ndarray | jax.numpy.ndarray
        Array cast to jax array if use_jax is True and jax is available. Input array returned otherwise.
    """
    if safe_autonomy_simulation.jax_available and use_jax and not isinstance(arr, jnp.ndarray):
        return jnp.array(arr)
    return arr
