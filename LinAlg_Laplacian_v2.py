import numpy as np
from scipy.sparse.linalg import spsolve
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


k_field = np.loadtxt("InputFolder/k_field_tiny.txt")
k_field_abs = np.abs(k_field)
h_field_initial = np.loadtxt("InputFolder/h_field_tiny.txt")
k_field_width = k_field.shape[1]
k_field_height = k_field.shape[0]
k_field_cnt = k_field_width * k_field_height

print(k_field)
print(k_field_abs)
print(h_field_initial)
print()

# Prepare k_vectors and h_vector
k_vector_signed = k_field.reshape(1, -1)
k_vector = k_field_abs.reshape(1, -1)
k_vector_up = shift_matrix(k_field_abs, "up").reshape(1, -1)
k_vector_down = shift_matrix(k_field_abs, "down").reshape(1, -1)
k_vector_left = shift_matrix(k_field_abs, "left").reshape(1, -1)
k_vector_right = shift_matrix(k_field_abs, "right").reshape(1, -1)
k_vector_sum = k_vector_up + k_vector_down + k_vector_left + k_vector_right
h_vector = h_field_initial.reshape(1, -1)

print(k_vector_signed)
print(k_vector)
print(k_vector_up)
print(k_vector_down)
print(k_vector_left)
print(k_vector_right)
print(k_vector_sum)
print(h_vector)
print()

# Prep laplace_operator_matrix
laplace_operator_matrix = np.zeros_like(k_vector)  # Seed matrix with empty array
for row in range(k_field_height):
    for col in range(k_field_width):
        # Prep a new row
        laplace_operator_row = np.zeros_like(k_vector)

        # Mark center cell
        index = row * k_field_width + col
        laplace_operator_row[0, index] = k_vector_sum[0, index]

        # Mark side cells
        if 0 <= (row + 1) <= (k_field_height - 1):
            index_up = (row + 1) * k_field_width + col
            laplace_operator_row[0, index_up] = - k_vector[0, index_up]
        if 0 <= (row - 1) <= (k_field_height - 1):
            index_down = (row - 1) * k_field_width + col
            laplace_operator_row[0, index_down] = - k_vector[0, index_down]
        if 0 <= (col - 1) <= (k_field_width - 1):
            index_right = row * k_field_width + (col - 1)
            laplace_operator_row[0, index_right] = - k_vector[0, index_right]
        if 0 <= (col + 1) <= (k_field_width - 1):
            index_left = row * k_field_width + (col + 1)
            laplace_operator_row[0, index_left] = - k_vector[0, index_left]

        # Append row to matrix
        laplace_operator_matrix = np.concatenate((laplace_operator_matrix, laplace_operator_row))

# Cut off empty top row
laplace_operator_matrix = np.delete(laplace_operator_matrix, np.arange(k_field_cnt)).reshape(-1, k_field_cnt)

print(laplace_operator_matrix)
print()


# Prep constant_head_A_matrix and constant_head_b_vector
constant_head_A_matrix = np.zeros_like(k_vector)
constant_head_vector = np.zeros((k_field_cnt, 1))
for cell in range(k_field_cnt):
    if k_vector_signed[0, cell] == -1:
        constant_head_A_row = np.zeros_like(k_vector)
        constant_head_A_row[0, cell] = 1
        laplace_operator_matrix[cell, :] = constant_head_A_row
        constant_head_vector[cell, 0] = h_vector[0, cell]

print(laplace_operator_matrix)
print(laplace_operator_matrix.shape)
print(constant_head_vector)
print()

h_vector_solved = spsolve(laplace_operator_matrix, constant_head_vector)
print(h_vector_solved)
