
import numpy as np

def type_changer(list1, to="list"):
    if to == "list":
        return list(list1)
    if to == "np":
        return np.array(list1)

def exact_2d_list(list1, list2):
    return np.array_equal(np.array(list1), np.array(list2))
