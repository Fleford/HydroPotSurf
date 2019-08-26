import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=150)

cosine_array = np.zeros((300, 300))
print(cosine_array)

m = 2   # Row
n = 4   # Column
m_max = cosine_array.shape[0]
n_max = cosine_array.shape[1]
for row in range(m_max):
    for col in range(n_max):
        # Calculate row component
        if m == 0:
            row_cosine = 1
        else:
            row_cosine = np.cos(np.pi / ((m_max) / m) * row)

        # Calculate col component
        if n == 0:
            col_cosine = 1
        else:
            col_cosine = np.cos(np.pi / ((n_max) / n) * col)

        cosine_array[row, col] = row_cosine * col_cosine

print(cosine_array)
plt.matshow(cosine_array)
plt.show()
