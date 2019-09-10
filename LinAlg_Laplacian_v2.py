import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import time
from numba import njit, prange
np.set_printoptions(linewidth=300)


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


# @njit()
def generate_laplace_operator_matrix(in_k_vector, in_k_vector_sum, in_k_field_height, in_k_field_width):
    # Prep laplace_operator_matrix
    laplace_op_matrix = np.zeros_like(in_k_vector)  # Seed matrix with empty array
    for row in range(in_k_field_height):
        for col in range(in_k_field_width):
            # Prep a new row
            laplace_operator_row = np.zeros_like(in_k_vector)

            # Prep index (Zero out row if k is zero)
            index = row * in_k_field_width + col

            if in_k_vector[0, index] == 0:
                laplace_op_matrix = np.concatenate((laplace_op_matrix, laplace_operator_row))
                continue

            # Mark center cell
            laplace_operator_row[0, index] = in_k_vector_sum[0, index]

            # Mark side cells
            if 0 <= (row + 1) <= (in_k_field_height - 1):
                index_up = (row + 1) * in_k_field_width + col
                laplace_operator_row[0, index_up] = - in_k_vector[0, index_up]
            if 0 <= (row - 1) <= (in_k_field_height - 1):
                index_down = (row - 1) * in_k_field_width + col
                laplace_operator_row[0, index_down] = - in_k_vector[0, index_down]
            if 0 <= (col - 1) <= (in_k_field_width - 1):
                index_right = row * in_k_field_width + (col - 1)
                laplace_operator_row[0, index_right] = - in_k_vector[0, index_right]
            if 0 <= (col + 1) <= (in_k_field_width - 1):
                index_left = row * in_k_field_width + (col + 1)
                laplace_operator_row[0, index_left] = - in_k_vector[0, index_left]

            # Append row to matrix
            laplace_op_matrix = np.concatenate((laplace_op_matrix, laplace_operator_row))

    # Cut off empty top row
    laplace_op_matrix = np.delete(laplace_op_matrix, np.arange(k_field_cnt)).reshape(-1, k_field_cnt)

    return laplace_op_matrix

print("hup")
# k_field = np.loadtxt("InputFolder/k_field_tiny.txt")
k_field = np.loadtxt("InputFolder/k_field_sv_230.txt")
k_field_abs = np.abs(k_field)
# h_field_initial = np.loadtxt("InputFolder/h_field_tiny.txt")
h_field_initial = np.loadtxt("InputFolder/h_field_sv_230.txt")

start_time = time.time()

k_field_width = k_field.shape[1]
k_field_height = k_field.shape[0]
k_field_cnt = k_field_width * k_field_height

# print(k_field)
# print(k_field_abs)
# print(h_field_initial)
# print()

print("hup")
# Prepare k_vectors and h_vector
k_vector_signed = k_field.reshape(1, -1)
k_vector = k_field_abs.reshape(1, -1)
k_vector_up = shift_matrix(k_field_abs, "up").reshape(1, -1)
k_vector_down = shift_matrix(k_field_abs, "down").reshape(1, -1)
k_vector_left = shift_matrix(k_field_abs, "left").reshape(1, -1)
k_vector_right = shift_matrix(k_field_abs, "right").reshape(1, -1)
k_vector_sum = k_vector_up + k_vector_down + k_vector_left + k_vector_right
h_vector = h_field_initial.reshape(1, -1)
print("hup")
# print(k_vector_signed)
# print(k_vector)
# print(k_vector_up)
# print(k_vector_down)
# print(k_vector_left)
# print(k_vector_right)
# print(k_vector_sum)
# print(h_vector)
# print()

# Prep laplace_operator_matrix
laplace_operator_matrix = generate_laplace_operator_matrix(k_vector, k_vector_sum, k_field_height, k_field_width)

print("hup")
# print(laplace_operator_matrix)
# print()

# Prep constant_head_A_matrix and constant_head_b_vector
constant_head_A_matrix = np.zeros_like(k_vector)
constant_head_vector = np.zeros((k_field_cnt, 1))
for cell in range(k_field_cnt):
    # Otherwise, revise rows and prep constant vector
    if k_vector_signed[0, cell] == -1:
        constant_head_A_row = np.zeros_like(k_vector)
        constant_head_A_row[0, cell] = 1
        laplace_operator_matrix[cell, :] = constant_head_A_row
        constant_head_vector[cell, 0] = h_vector[0, cell]
print("hup")
# Clear out zero rows and columns
# print(laplace_operator_matrix)
# print(laplace_operator_matrix.shape)
# print(constant_head_vector)
data = np.concatenate((laplace_operator_matrix, constant_head_vector), axis=1)
# print(data)
data = np.delete(data, np.where(~data.any(axis=1))[0], axis=0)  # remove zero rows
data = np.delete(data, np.where(~data.any(axis=0))[0], axis=1)  # remove zero columns
laplace_operator_matrix = data[:, 0:-1]
constant_head_vector = data[:, -1].reshape(-1, 1)
# print(laplace_operator_matrix)
# print(constant_head_vector)
# print()

print("Seconds to prep matrices: " + str(time.time() - start_time))
start_time = time.time()

# Solve matrix
sp_laplace_operator_matrix = csc_matrix(laplace_operator_matrix)
sp_constant_head_vector = csc_matrix(constant_head_vector)
h_vector_solved = spsolve(sp_laplace_operator_matrix, sp_constant_head_vector)

print("Seconds to solve matrices: " + str(time.time() - start_time))
start_time = time.time()

# Unravel solution
h_field = np.zeros_like(k_vector)
index_solution = 0
for index_field in range(k_field_cnt):
    if k_vector[0, index_field] != 0:
        h_field[0, index_field] = h_vector_solved[index_solution]
        index_solution += 1
h_field = h_field.reshape(k_field.shape)
print("Seconds to unravel matrices: " + str(time.time() - start_time))
start_time = time.time()

print(h_field)
plt.matshow(h_field)
plt.show()
