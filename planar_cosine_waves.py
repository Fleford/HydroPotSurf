import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=150)


def generate_cosine_array(array_like, m_row_half_wavelengths, n_col_half_wavelengths):
    # Number of half-wavelengths along the Row
    # Number of half-wavelengths along the Column
    m_max = array_like.shape[0]
    n_max = array_like.shape[1]
    basis_array = np.zeros_like(array_like)

    for row in range(m_max):
        for col in range(n_max):
            # Calculate row component
            if m_row_half_wavelengths == 0:
                row_cosine = 1
            else:
                row_cosine = np.cos(np.pi / (m_max / m_row_half_wavelengths) * row)

            # Calculate column component
            if n_col_half_wavelengths == 0:
                col_cosine = 1
            else:
                col_cosine = np.cos(np.pi / (n_max / n_col_half_wavelengths) * col)

            basis_array[row, col] = row_cosine * col_cosine
    return basis_array


if __name__ == "__main__":
    matrix = np.zeros((300, 300))
    print(matrix)

    adjustment_array = generate_cosine_array(matrix, 1, 0)
    print(adjustment_array)

    plt.matshow(adjustment_array)
    plt.show()
