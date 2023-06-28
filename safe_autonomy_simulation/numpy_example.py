'''numpy module docstring'''
import numpy as np


def array_add():
    '''arrary add function docstring'''
    nums = np.array([1, 2, 3, 4, 5])
    nums2 = nums + 1
    return nums2


if __name__ == '__main__':
    print(array_add())
