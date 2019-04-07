import numpy as np
from scipy.linalg import lstsq
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
    k_matrix = np.absolute(k_matrix)
    hk_matrix = h_matrix * k_matrix
    # When dividing by zero, a zero is returned
    new_hk_matrix = np.divide(shift_matrix_sum(hk_matrix), shift_matrix_sum(k_matrix),
                              out=np.zeros_like(shift_matrix_sum(hk_matrix)), where=shift_matrix_sum(k_matrix) != 0)
    # # Zero out regions with zero hk
    new_hk_matrix = np.divide((new_hk_matrix * k_matrix), k_matrix,
                              out=np.zeros_like(new_hk_matrix * hk_matrix), where=k_matrix != 0)
    return new_hk_matrix


def laplace_smooth_iter(h_const, h_matrix, k_matrix, convergence_threshold=0):
    # Performs a number of iterations of Laplace smoothing
    # h_matrix = np.ones_like(h_matrix) * h_field.max()
    while True:
        # Make masks for constant heads
        zero_at_const_h = np.ma.masked_equal(h_const, 0).mask * 1
        one_at_const_h = np.ma.masked_not_equal(h_const, 0).mask * 1
        # Perform smoothing
        new_h_matrix = laplace_smooth(h_matrix, k_matrix)
        # Apply constants
        new_h_matrix = zero_at_const_h * new_h_matrix + one_at_const_h * h_const
        # Zero out const heads on zero k
        new_h_matrix = np.divide((new_h_matrix * k_matrix), k_matrix,
                                 out=np.zeros_like(new_h_matrix * k_matrix), where=k_matrix != 0)
        # Stop when the solution stops changing:
        max_diff = np.max(new_h_matrix - h_matrix)
        print(max_diff)
        if max_diff <= convergence_threshold:
            break

        # Update matrix
        h_matrix = new_h_matrix
    return h_matrix


h_field = np.loadtxt("InputFolder/initial_heads.txt")
k_field = np.loadtxt("InputFolder/k_field.txt")
# h_field = np.ones((10, 20))*10

# Fit a plane to data
h_obs = np.array([[0, 0, 15],
                  [3, 12, 11],
                  [6, 9, 12],
                  [8, 18, 1]])
# print(h_obs[:, 2].reshape(-1, 1))
# print(np.ones_like(h_obs[:, 2]).reshape(-1, 1))
# print(h_obs[:, 0:2])
# print(np.concatenate((h_obs[:, 0:2], np.ones_like(h_obs[:, 2]).reshape(-1, 1)), axis=1))
M = np.concatenate((h_obs[:, 0:2], np.ones_like(h_obs[:, 2]).reshape(-1, 1)), axis=1)
y = h_obs[:, 2].reshape(-1, 1)
abc, res, rnk, s = lstsq(M, y)
# print(M)
# print(y)
# print(abc)
# print(M.dot(abc))

# Fill matrix with plane data
y = np.arange(0, 10)
x = np.arange(0, 20)
x_index, y_index = np.meshgrid(x, y)
# print(y_index)
# print(x_index)
m_grid = np.concatenate((y_index.reshape(-1, 1), x_index.reshape(-1, 1)), axis=1)
m_grid = np.concatenate((m_grid, np.ones_like(y_index.reshape(-1, 1))), axis=1)
# print(m_grid)
h_plane = m_grid.dot(abc).reshape(10, 20)
plt.matshow(h_plane)


# Test Laplace smoother
plt.matshow(h_field)
plt.matshow(k_field)
plt.matshow(laplace_smooth_iter(h_field, h_field, k_field))
plt.show()
