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


def laplace_smooth_iter(h_matrix, k_matrix, convergence_threshold=0.001):
    # Performs a number of iterations of Laplace smoothing
    # h_matrix = np.ones_like(h_matrix) * h_field.max()

    # Make mask and copy of initial h_matrix for constant heads (where k is negative)
    one_at_neg_k = np.ma.masked_less(k_matrix, 0).mask * 1
    zero_at_neg_k = np.ma.masked_greater_equal(k_matrix, 0).mask * 1
    initial_h_matrix = h_matrix

    # Sanitize k_matrix from negatives
    k_matrix_abs = np.absolute(k_matrix)

    while True:
        # Perform smoothing
        new_h_matrix = laplace_smooth(h_matrix, k_matrix_abs)
        # Apply constant heads (where k is negative)
        new_h_matrix = zero_at_neg_k * new_h_matrix + one_at_neg_k * initial_h_matrix
        # Zero out const heads on zero k
        new_h_matrix = np.divide((new_h_matrix * k_matrix_abs), k_matrix_abs,
                                 out=np.zeros_like(new_h_matrix * k_matrix_abs), where=k_matrix_abs != 0)
        # Calculate change
        max_diff = np.max(new_h_matrix - h_matrix)

        # Update matrix
        h_matrix = new_h_matrix

        # Stop if the change is too small
        print(max_diff)
        if max_diff <= convergence_threshold:
            break

    return h_matrix


# Load in observation values
obs_field = np.loadtxt("InputFolder/initial_heads.txt")
h_field = obs_field
# h_field = np.ones((10, 20))*10
# Load k_field with parameters
k_field = np.loadtxt("InputFolder/k_field.txt")
k_field2 = np.loadtxt("InputFolder/k_field2.txt")
# print(k_field)
# k_field = np.absolute(k_field)
one_at_neg_k = np.ma.masked_less(k_field, 0).mask*1
zero_at_neg_k = np.ma.masked_greater_equal(k_field, 0).mask*1
# print(one_at_neg_k)
# print(zero_at_neg_k)

# Convert observation matrix to observation matrix
zero_at_obs_h = np.ma.masked_equal(obs_field, 0).mask * 1
one_at_obs_h = np.ma.masked_not_equal(obs_field, 0).mask * 1
print(zero_at_obs_h)
print(one_at_obs_h)
obs_args = np.argwhere(obs_field > 0)
obs_vals = obs_field[obs_args[:, 0], obs_args[:, 1]].reshape(-1, 1)
obs_mat = np.concatenate((obs_args, obs_vals), axis=1)
print(obs_args)
print(obs_vals)
print(obs_mat)

# Fit a plane to points
h_obs = np.array([[0, 0, 15],
                  [3, 12, 11],
                  [6, 9, 12],
                  [8, 18, 1]])
h_obs = obs_mat
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
# plt.matshow(h_plane)


# Test Laplace smoother
# plt.matshow(obs_field)
# plt.matshow(h_plane)
# plt.matshow(k_field)
# plt.matshow(laplace_smooth_iter(h_plane, k_field))
# plt.contour(laplace_smooth_iter(h_plane, k_field))
result = laplace_smooth_iter(obs_field, k_field)
result2 = laplace_smooth_iter(result, k_field2)
# plt.matshow(result2)
diff = result2 - result
# plt.matshow(diff)
new_obs = obs_field-diff
result3 = laplace_smooth_iter(new_obs, k_field)
result4 = laplace_smooth_iter(result3, k_field2)
# plt.matshow(result4)
diff2 = result4 - result
# plt.matshow(diff2)
new_obs_2 = new_obs-diff2
result5 = laplace_smooth_iter(new_obs_2, k_field)
result6 = laplace_smooth_iter(result5, k_field2)
plt.matshow(result6)
diff3 = result6 - result
plt.matshow(diff3)
result6[result6 == 0] = np.nan
plt.contour(result6)
plt.show()
