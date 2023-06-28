'''Unit tests'''

import numpy as np
from safe_autonomy_simulation.numpy_example import array_add


def test_answer():
    nums = np.array([3, 4, 5, 6, 7])
    assert array_add().all() == nums.all()
