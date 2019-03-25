import numpy as np
import matplotlib.pyplot as plt


def shift_matrix(matrix, direction):
    # Shifts a 2d matrix in a direction and pads with zeros
    if direction == "up":
        return np.delete(np.pad(matrix, ((0, 1), (0, 0)), 'constant'), 0, 0)
    elif direction == "down":
        return np.delete(np.pad(matrix, ((1, 0), (0, 0)), 'constant'), -1, 0)
    elif direction == "left":
        return np.delete(np.pad(matrix, ((0, 0), (0, 1)), 'constant'), 0, 1)
    elif direction == "right":
        return np.delete(np.pad(matrix, ((0, 0), (1, 0)), 'constant'), -1, 1)
    else:
        return "NO DIRECTION SPECIFIED: up, down, left, right"


def shift_matrix_sum(matrix):
    # Each 2d element becomes the sum of all its adjacent elements
    # Empty cells are padded with zero
    return shift_matrix(matrix, "up") + shift_matrix(matrix, "down")\
               + shift_matrix(matrix, "left") + shift_matrix(matrix, "right")


def laplace_smooth(h_matrix, k_matrix):
    # Performs one iteration of the Laplace smoothing operation
    hk_matrix = h_matrix * k_matrix
    # When dividing by zero, a zero is returned
    return np.divide(shift_matrix_sum(hk_matrix), shift_matrix_sum(k_matrix),
                     out=np.zeros_like(shift_matrix_sum(hk_matrix)), where=shift_matrix_sum(k_matrix) != 0)


def laplace_smooth_iter(h_matrix, k_matrix, iterations):
    # Performs a number of iterations of Laplace smoothing
    for runs in range(iterations):
        h_matrix = laplace_smooth(h_matrix, k_matrix)
    return h_matrix


h_field = np.loadtxt("InputFolder/initial_heads.txt")
k_field = np.loadtxt("InputFolder/k_field.txt")
# k_field = np.ones((10, 20))

# hk_field = h_field * k_field
#
# hk_sum_field = shift_matrix_sum(hk_field)
# k_sum_field = shift_matrix_sum(k_field)
# new_h_field = hk_sum_field / k_sum_field

print(h_field)
print(k_field)
print(np.nan_to_num(laplace_smooth(h_field, k_field)))

plt.matshow(h_field)
plt.matshow(k_field)
plt.matshow(laplace_smooth_iter(h_field, k_field, 160))
plt.show()
