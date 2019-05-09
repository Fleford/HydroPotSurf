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
    initial_h_matrix = h_matrix.copy()

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
        # print(max_diff)
        if max_diff <= convergence_threshold:
            break

    return h_matrix


def calculate_boundary_values(obs_matrix, k_cnst_obs, k_cnst_bnd):
    # Estimates the head values at the constant-head boundary
    # obs_matrix = Contains observation heads (zero at all other points)
    # k_cnst_obs = K field with -1 at constant-head boundary
    # k_cnst_bnd = K field with -1 at observation heads

    # Make initial specified observation heads as a copy of obs_matrix
    spcfd_obs_h = obs_matrix.copy()

    # Calculate constant-head boundary values, given specified observation heads
    h_field_cnst_obs = laplace_smooth_iter(spcfd_obs_h, k_cnst_obs)

    # Calculate observation values, given constant-head boundary values
    h_field_cnst_bnd = laplace_smooth_iter(h_field_cnst_obs, k_cnst_bnd)

    # Make mask for observation heads (1 only at observation heads)
    obs_h_mask = k_cnst_obs.copy()
    obs_h_mask[obs_h_mask != -1] = 0
    obs_h_mask[obs_h_mask == -1] = 1
    # print(obs_h_mask)

    # Cut out estimated observation heads from h_field_cnst_bnd
    obs_h_from_bnds = h_field_cnst_bnd * obs_h_mask
    obs_h_from_bnds = obs_h_from_bnds[obs_h_from_bnds != 0]
    obs_h_from_bnds = obs_h_from_bnds.reshape(-1, 1)
    # print(obs_h_from_bnds)

    # Cut out measured observation heads from obs_matrix
    obs_h_from_meas = obs_matrix * obs_h_mask
    obs_h_from_meas = obs_h_from_meas[obs_h_from_meas != 0]
    obs_h_from_meas = obs_h_from_meas.reshape(-1, 1)
    # print(obs_h_from_meas)

    # Find scale and offset values to fit calculated heads to observed heads (mx = y)
    est_obs_matrix = np.concatenate((obs_h_from_bnds, np.ones_like(obs_h_from_bnds)), axis=1)
    # print(est_obs_matrix)
    scale_offset, res, rnk, s = lstsq(est_obs_matrix, obs_h_from_meas)
    # print(scale_offset)
    # print(est_obs_matrix.dot(scale_offset))

    # Apply scale and offset values to h_field_cnst_bnd
    h_field_cnst_bnd = scale_offset[0]*h_field_cnst_bnd + scale_offset[1]

    # Set the heads at no flow regions to zero
    zero_k_mask = k_cnst_bnd.copy()
    zero_k_mask[zero_k_mask != 0] = 1
    # print(zero_k_mask)
    h_field_cnst_bnd = h_field_cnst_bnd * zero_k_mask

    # Return your hardwork
    return h_field_cnst_bnd


def above_below_pivot_masks(h_matrix, pivot_value, k_matrix):
    # Given an h_field and the pivot value, it generates two masks
    # for cells with heads above and below the pivot value

    k_mask = np.ma.masked_not_equal(k_matrix, 0).mask * 1
    above_pivot = np.ma.masked_greater(h_matrix, pivot_value).mask * 1 * k_mask
    below_or_equal_pivot = np.ma.masked_less_equal(h_matrix, pivot_value).mask * 1 * k_mask

    return above_pivot, below_or_equal_pivot


def safe_divide(dividend, divisor):
    # Prior to dividing, all zeros in the divisor are set to one
    divisor_oned = divisor.copy()
    divisor_oned[divisor_oned == 0] = 1

    return dividend / divisor_oned


def split_into_sign_and_magnitude(matrix):
    # Splits matrix of values into signs (-1,0,1) and magnitudes
    # Multiplying the two results yields the original value

    # Create magnitude matrix
    matrix_magnitude = matrix.copy()
    matrix_magnitude = np.absolute(matrix_magnitude)

    # Create sign matrix
    matrix_sign = matrix.copy()
    matrix_sign = safe_divide(matrix_sign, matrix_magnitude)

    return matrix_sign, matrix_magnitude



# Make function that converts a single input matrix into multiple gw_model variables




# Load in observation values
obs_field = np.loadtxt("InputFolder/initial_heads.txt")

h_field = obs_field
# h_field = np.ones((10, 20))*10
# Load k_field with parameters
k_field_const_obs = np.loadtxt("InputFolder/k_field.txt")
k_field2 = np.loadtxt("InputFolder/k_field2.txt")
# k_field = np.absolute(k_field)
one_at_neg_k = np.ma.masked_less(k_field_const_obs, 0).mask*1
zero_at_neg_k = np.ma.masked_greater_equal(k_field_const_obs, 0).mask*1
# print(one_at_neg_k)
# print(zero_at_neg_k)


# Test Laplace smoother
# plt.matshow(obs_field)
# plt.matshow(h_plane)
# plt.matshow(k_field)
# plt.matshow(laplace_smooth_iter(h_plane, k_field))
# plt.contour(laplace_smooth_iter(h_plane, k_field))

initial_h_field = calculate_boundary_values(obs_field, k_field_const_obs, k_field2)
# levels = np.arange(np.amin(initial_h_field), np.amax(initial_h_field), 0.5)
# initial_h_field[initial_h_field == 0] = np.nan
# empty = np.zeros_like(initial_h_field)
# plt.matshow(initial_h_field)
# plt.matshow(obs_field)
# plt.contour(initial_h_field, levels=levels)
# plt.show()

# Create masks for cells above and below pivot head
# print(initial_h_field)
k_mask = np.ma.masked_equal(k_field_const_obs, -1).mask*1
# print(k_mask)
obs_ind = np.nonzero(k_mask)
h_pivot = initial_h_field[obs_ind[0][0], obs_ind[1][0]]
# print(h_pivot)
# above_h_cell = np.ma.masked_greater(initial_h_field, 10.1).mask*1*k_mask
# belequal_h_cell = np.ma.masked_less_equal(initial_h_field, 10.1).mask*1*k_mask

# above_h_cell, belequal_h_cell = above_below_pivot_masks(initial_h_field, h_pivot, k_field)

# print(above_h_cell)
# print(belequal_h_cell)


# Create list of indexes for each observation head
print(k_mask)
pivots = np.argwhere(k_mask)
# print(pivots)

# grab head of pivot cell for calculated and observed head. Calculate difference
print(pivots[0])
pivot_new_h_field = initial_h_field[pivots[0][0], pivots[0][1]]
pivot_obs_field = obs_field[pivots[0][0], pivots[0][1]]
print(pivot_new_h_field)
print(pivot_obs_field)
diff = pivot_new_h_field - pivot_obs_field
print(diff)
# Scale diff with min and max of surface
scaled_diff = diff/(initial_h_field.max() - initial_h_field.min())
print(scaled_diff)

# create the delta_k array (apply all pivots into one delta matrix)
above_h_cell, belequal_h_cell = above_below_pivot_masks(initial_h_field, pivot_new_h_field, k_field_const_obs)
k_zero_mask = above_h_cell + belequal_h_cell
print(above_h_cell)
print(belequal_h_cell)
delta_k = (above_h_cell * -1 + belequal_h_cell * 1) * scaled_diff * k_zero_mask
print(delta_k)


# apply delta_k to k field and rescale k field
delta_k = delta_k - delta_k.min() * k_zero_mask     # offset delta_k (no negative k)
k_field2_sign, k_field2_mag = split_into_sign_and_magnitude(k_field2)
k_field2_mag = k_field2_mag + delta_k
# print(k_field2_mag)
k_field2_mag = k_field2_mag / np.ma.masked_equal(k_field2_mag, 0).min()
k_field2_new = k_field2_sign * k_field2_mag
print(k_field2_new)

# run gw_model with old and new k_field
h_with_old_k = laplace_smooth_iter(initial_h_field, k_field2)
h_with_new_k = laplace_smooth_iter(initial_h_field, k_field2_new)
print(h_with_old_k)
print(h_with_new_k)
# plt.matshow(h_with_new_k - h_with_old_k)
# plt.show()

# Check errors
pivot_new_h_field = h_with_new_k[pivots[0][0], pivots[0][1]]
pivot_obs_field = obs_field[pivots[0][0], pivots[0][1]]
print(pivot_new_h_field)
print(pivot_obs_field)
diff = pivot_new_h_field - pivot_obs_field
print(diff)