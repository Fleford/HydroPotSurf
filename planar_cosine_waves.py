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


def diagonal_counter(end_cnt):
    m = 0   # Row
    n = 0   # Column
    cnt = 0

    if cnt == end_cnt:
        print(m, n)
        return m, n

    m = 0   # Row
    n = 1   # Column

    cnt += 1
    if cnt == end_cnt:
        print(m, n)
        return m, n

    while True:
        # Slosh
        m_done = n
        n_done = m
        if n < m:
            reverse = True
        else:
            reverse = False
        while m_done != m and n_done != n:
            if reverse:
                m -= 1
                n += 1
            else:
                m += 1
                n -= 1

            cnt += 1
            if cnt == end_cnt:
                print(m, n)
                return m, n

        # Increment
        if m != 0:
            m += 1
        if n != 0:
            n += 1

        cnt += 1
        if cnt == end_cnt:
            print(m, n)
            return m, n


if __name__ == "__main__":
    # matrix = np.zeros((300, 300))
    # print(matrix)
    #
    # adjustment_array = generate_cosine_array(matrix, 2, 3)
    # print(adjustment_array)
    #
    # plt.matshow(adjustment_array)
    # plt.show()
    for x in range(15):
        diagonal_counter(x)
