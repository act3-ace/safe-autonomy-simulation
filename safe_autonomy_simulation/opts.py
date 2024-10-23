"""Check for package options set using env variables"""

import os


def jax_available():
    """Whether or not JAX is available for numerical computing and JIT compilation

    Checks the environment for the JAX package. Also checks the environment
    variable `USE_JAX` to see if JAX should be used for
    numerical computing and JIT compilation. If the JAX package is available
    and `USE_JAX = 1` then this function returns True. Otherwise this
    function returns False.

    Returns
    -------
    bool
        whether or not JAX is available for numerical computing and JIT compilation
    """
    if "USE_JAX" in os.environ:
        assert (
            os.environ["USE_JAX"] in ["0", "1"]
        ), f'invalid value {os.environ["USE_JAX"]} for environment variable USE_JAX. Must be "0" or "1"'
        result = bool(int(os.environ["USE_JAX"]))
        if result:
            try:
                import jax
            except ImportError:
                raise ImportError(
                    "Environment variable `USE_JAX = 1` but JAX is not found in environment. JAX can be installed via `pip install safe-autonomy-simulation[jax]`"
                )
        return result
    else:
        try:
            import jax
        except ImportError:
            return False
        return True
