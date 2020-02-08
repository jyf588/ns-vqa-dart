import numpy as np


def up_vector_to_rotation(up_vector):
    rotation_matrix = np.eye(3)
    rotation_matrix[:, -1] = up_vector
    rotation_matrix = list(rotation_matrix.reshape((9,)))
    return rotation_matrix


def vec_to_str(vec):
    list_of_str = [f"{float(v):.2f}" for v in vec]
    comma_sep_str = ", ".join(list_of_str)
    return f"({comma_sep_str})"
