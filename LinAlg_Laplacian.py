import numpy as np
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
h_field_const = np.loadtxt("InputFolder/h_field_tiny.txt")
k_field_width = k_field.shape[1]
k_field_height = k_field.shape[0]
k_field_cnt = k_field_width * k_field_height

print(k_field)
print(h_field_const)
print(k_field_width)
print(k_field_height)

# Prepare k_vectors
k_vector = k_field.reshape(1, -1)
k_vector_up = shift_matrix(k_field, "up").reshape(1, -1)
k_vector_down = shift_matrix(k_field, "down").reshape(1, -1)
k_vector_left = shift_matrix(k_field, "left").reshape(1, -1)
k_vector_right = shift_matrix(k_field, "right").reshape(1, -1)
k_vector_sum = k_vector_up + k_vector_down + k_vector_left + k_vector_right

print(k_vector)
print(k_vector_up)
print(k_vector_down)
print(k_vector_left)
print(k_vector_right)
print(k_vector_sum)
print()

laplace_operator_row = np.zeros_like(k_vector)
laplace_operator_matrix = laplace_operator_row.copy()   # Seed matrix with empty array
row, col = 2, 0
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

laplace_operator_matrix = np.concatenate((laplace_operator_matrix, laplace_operator_row))
laplace_operator_matrix = np.concatenate((laplace_operator_matrix, laplace_operator_row))

# Cut off empty top row
laplace_operator_matrix = np.delete(laplace_operator_matrix, np.arange(k_field_cnt)).reshape(-1, k_field_cnt)

print(laplace_operator_matrix)
