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
    # k_matrix used just to determine active cells

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


def calculate_new_k_field(h_matrix, k_matrix, obs_matrix):
    # Calculates a slight better k_matrix
    # Obs_matrix = zero at all values other than observation values

    # Build index list for all observed heads
    obs_indexes = np.argwhere(obs_matrix)
    # print(obs_indexes)

    # Prepare an empty delta_k matrix
    total_delta_k = np.zeros_like(k_matrix)

    # Prepare mask for zero k
    zero_k_mask = np.ma.masked_not_equal(k_matrix, 0).mask * 1

    # For each observation,
    for obs_index in obs_indexes:
        # Calculate difference between calculated and observed head
        h_calculated = h_matrix[obs_index[0], obs_index[1]]
        h_observed = obs_field[obs_index[0], obs_index[1]]
        h_diff = h_calculated - h_observed

        # Scale diff with min and max of h_field
        h_diff_scaled = h_diff / (h_matrix.max() - h_matrix.min())

        # Create delta_k array
        above_h_cal, below_or_equal_h_cal = above_below_pivot_masks(h_matrix, h_calculated, k_matrix)
        delta_k = (above_h_cal * -1 + below_or_equal_h_cal * 1) * h_diff_scaled * zero_k_mask

        # Apply delta_k to the overall_delta_k
        total_delta_k = total_delta_k + delta_k

    # Offset the delta_k so there are no negatives
    offset_total_delta_k = (total_delta_k - total_delta_k.min()) * zero_k_mask

    # Apply the total delta k to the k field
    k_matrix_sign, k_matrix_mag = split_into_sign_and_magnitude(k_matrix)
    k_matrix_mag = k_matrix_mag + offset_total_delta_k
    k_matrix_mag = k_matrix_mag / np.ma.masked_equal(k_matrix_mag, 0).min()  # Scale k so that min = 1
    k_matrix_new = k_matrix_sign * k_matrix_mag

    return k_matrix_new


def input_matrix_to_parameter_matrices(input_matrix):
    # Function for converting initial_input matrix into multiple gw parameter matrices

    # Given x as an element within the input matrix:
    # (x < 0) = constant-head cell
    # (x = 0) = inactive cell
    # (x = 1) = active cells
    # (x > 1) = observed head (assumed an active cell)

    # k_field_cnst_bnd
    k_field_cnst_bnd = input_matrix.copy()
    k_field_cnst_bnd = np.clip(k_field_cnst_bnd, -1, 1)
    # print(k_field_cnst_bnd)

    # k_field_cnst_obs
    k_field_cnst_obs = input_matrix.copy()
    k_field_cnst_obs = np.absolute(k_field_cnst_obs)
    k_field_cnst_obs[k_field_cnst_obs > 1] = -1
    # print(k_field_cnst_obs)

    # obs_field
    obs_field = input_matrix.copy()
    obs_field[obs_field <= 1] = 0
    # print(obs_field)

    return k_field_cnst_bnd, k_field_cnst_obs, obs_field


# Load in initial input file
initial_input = np.loadtxt("InputFolder/initial_input.txt")
k_field_const_bnd, k_field_const_obs, obs_field = input_matrix_to_parameter_matrices(initial_input)

h_field = calculate_boundary_values(obs_field, k_field_const_obs, k_field_const_bnd)
# levels = np.arange(np.amin(h_field), np.amax(h_field), 0.5)
# h_field[h_field == 0] = np.nan
# empty = np.zeros_like(h_field)
# plt.matshow(h_field)
# plt.matshow(obs_field)
# plt.contour(h_field, levels=levels)
# plt.show()

# Calculate new k
k_field2_new = calculate_new_k_field(h_field, k_field_const_bnd, obs_field)

# run gw_model with old and new k_field
h_with_old_k = laplace_smooth_iter(h_field, k_field_const_bnd)
h_with_new_k = laplace_smooth_iter(h_field, k_field2_new)
# print(h_with_old_k)
# print(h_with_new_k)
# plt.matshow(h_with_new_k - h_with_old_k)
# plt.show()