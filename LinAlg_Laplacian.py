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

print(k_field)
print(h_field_const)
print(k_field_width)

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
print(laplace_operator_row)
row, col = 0, 0
index = row * k_field_width + col
index_up = row * k_field_width + col



