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


def adjust_h_field_to_fit_obs(h_matrix, obs_matrix, obs_mask, k_mask):
    # Adjusts the h_field using linear transformations to fit data
    # h_matrix = The h_field to be adjusted
    # obs_matrix = Contains obs heads at certain cells. Remaining cells are set to zero
    # obs_mask = Matrix with 1 at all cells with obs data, otherwise 0
    # k_mask  = Matrix with 1 at nonzero k, else 0

    # Cut out estimated observation heads from h_matrix
    obs_h_from_h_matrix = h_matrix * obs_mask
    obs_h_from_h_matrix = obs_h_from_h_matrix[obs_h_from_h_matrix != 0]
    obs_h_from_h_matrix = obs_h_from_h_matrix.reshape(-1, 1)
    # print(obs_h_from_h_matrix)

    # Cut out measured observation heads from obs_matrix
    obs_h_from_obs_matrix = obs_matrix * obs_mask
    obs_h_from_obs_matrix = obs_h_from_obs_matrix[obs_h_from_obs_matrix != 0]
    obs_h_from_obs_matrix = obs_h_from_obs_matrix.reshape(-1, 1)
    # print(obs_h_from_obs_matrix)

    # Find scale and offset values to fit calculated heads to observed heads (mx = y)
    est_obs_matrix = np.concatenate((obs_h_from_h_matrix, np.ones_like(obs_h_from_h_matrix)), axis=1)
    # print(est_obs_matrix)
    scale_offset, res, rnk, s = lstsq(est_obs_matrix, obs_h_from_obs_matrix)
    # print(scale_offset)
    # print(est_obs_matrix.dot(scale_offset))

    # Apply scale and offset values to h_field_cnst_bnd
    h_field_adjusted = scale_offset[0]*h_matrix + scale_offset[1]

    # Set the heads at no flow regions to zero
    h_field_adjusted = h_field_adjusted * k_mask

    return h_field_adjusted


def calculate_boundary_values(obs_matrix, k_cnst_obs, k_cnst_bnd, convergence_threshold=0.001):
    # Estimates the head values at the constant-head boundary
    # obs_matrix = Contains observation heads (zero at all other points)
    # k_cnst_obs = K field with -1 at constant-head boundary
    # k_cnst_bnd = K field with -1 at observation heads

    # Make initial specified observation heads as a copy of obs_matrix
    spcfd_obs_h = obs_matrix.copy()

    # Make mask for observation heads (1 only at observation heads)
    obs_h_mask = k_cnst_obs.copy()
    obs_h_mask[obs_h_mask != -1] = 0
    obs_h_mask[obs_h_mask == -1] = 1
    # print(obs_h_mask)

    # Make mask for zero k cells (1 at non-zero k, else 0)
    zero_k_mask = k_cnst_bnd.copy()
    zero_k_mask[zero_k_mask != 0] = 1
    # print(zero_k_mask)

    # Make invert k_field (swap roles of active and specified cells)
    k_cnst_bnd_inverted = k_cnst_bnd * -1

    # Calculate constant-head boundary values, given specified observation heads
    h_field_cnst_obs = laplace_smooth_iter(spcfd_obs_h, k_cnst_obs)

    # Calculate observation values, given constant-head boundary values
    h_field_cnst_bnd = laplace_smooth_iter(h_field_cnst_obs, k_cnst_bnd)

    # Fit h field to observed heads
    h_field_cnst_bnd = adjust_h_field_to_fit_obs(h_field_cnst_bnd, obs_matrix, obs_h_mask, zero_k_mask)

    # Setup previous error variable
    prev_max_diff = np.inf

    # Start Loop
    while True:
        # Copy an unmodified version of the h field
        h_field_cnst_bnd_old = h_field_cnst_bnd.copy()

        # Calculate constant-head boundary values, given inverted k field
        h_field_cnst_bnd = laplace_smooth_iter(h_field_cnst_bnd, k_cnst_bnd_inverted)

        # Calculate h field, given new boundary values
        h_field_cnst_bnd = laplace_smooth_iter(h_field_cnst_bnd, k_cnst_bnd)

        # Fit h field to observed heads
        h_field_cnst_bnd = adjust_h_field_to_fit_obs(h_field_cnst_bnd, obs_matrix, obs_h_mask, zero_k_mask)

        # Calculate max change of h field
        max_diff = np.max(h_field_cnst_bnd - h_field_cnst_bnd_old)

        # Stop if the change is too small
        print(max_diff)
        if max_diff > prev_max_diff:
            break

        # Update error
        prev_max_diff = max_diff

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


def calculate_new_k_field(h_matrix, k_matrix, obs_matrix, h_matrix_smooth):
    # Calculates a slightly better k_matrix
    # Obs_matrix = zero at all values other than observation values
    # h_matrix_smooth = first h_matrix made. Usually the smoothest. Used for pivot_mask making

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
        h_pivot = h_matrix_smooth[obs_index[0], obs_index[1]]
        h_calculated = h_matrix[obs_index[0], obs_index[1]]
        h_observed = obs_field[obs_index[0], obs_index[1]]
        h_diff = h_calculated - h_observed

        # Scale diff with min and max of h_field
        h_diff_scaled = h_diff / (h_matrix.max() - h_matrix.min())

        # Create delta_k array
        above_h_cal, below_or_equal_h_cal = above_below_pivot_masks(h_matrix_smooth, h_pivot, k_matrix)
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


def iteratively_adjust_k(h_matrix, k_matrix, obs_matrix):
    # Iteratively adjusts the k_field until convergence is reached

    # Assume provided h_matrix is the smoothest
    # Copy the original h_matrix
    h_matrix_smooth = h_matrix.copy()

    # Setup previous error variable
    previous_error = np.inf

    for x in range(10000):
        # Calculate new k matrix
        k_matrix_new = calculate_new_k_field(h_matrix, k_matrix, obs_matrix, h_matrix_smooth)

        # Calculate new h matrix
        h_matrix_new = laplace_smooth_iter(h_matrix, k_matrix_new)

        # Calculate maximum error
        h_obs_mask = obs_matrix.copy()
        h_obs_mask[h_obs_mask != 0] = 1
        h_error = (h_matrix_new - obs_matrix) * h_obs_mask
        h_error_abs = np.absolute(h_error)
        print(h_error_abs.max())
        if h_error_abs.max() > previous_error:
            break

        # Replace current k and h matrix
        k_matrix = k_matrix_new
        h_matrix = h_matrix_new

        # Update error
        previous_error = h_error_abs.max()

    return h_matrix, k_matrix


# Load in matrices
initial_input = np.loadtxt("InputFolder/initial_input.txt")
k_field, k_field_const_obs, obs_field = input_matrix_to_parameter_matrices(initial_input)

# Calculate bnd heads and initial h field
h_field = calculate_boundary_values(obs_field, k_field_const_obs, k_field)

# # Calculate new k
# k_field2_new = calculate_new_k_field(h_field, k_field, obs_field, h_field)
#
# # run gw_model with old and new k_field
# h_with_old_k = laplace_smooth_iter(h_field, k_field)
# h_with_new_k = laplace_smooth_iter(h_field, k_field2_new)
# print(h_with_old_k)
# print(h_with_new_k)
# plt.matshow(h_with_new_k - h_with_old_k)
# plt.show()

# # Iteratively adjust k
# new_h_field, new_k_field = iteratively_adjust_k(h_field, k_field, obs_field)
# print("hup!")
# print(new_h_field)
# print(new_k_field)

# Display current results
levels = np.arange(np.amin(h_field), np.amax(h_field), 0.5)
plt.matshow(h_field)
# plt.matshow(new_h_field)
# plt.matshow(new_h_field - obs_field)

h_field[h_field == 0] = np.nan
plt.matshow(k_field)
plt.contour(h_field, levels=levels)

# plt.matshow(new_k_field)
# # empty = np.zeros_like(new_h_field)
# # plt.matshow(empty, cmap="gray")
# new_h_field[new_h_field == 0] = np.nan
# plt.contour(new_h_field, levels=levels)

plt.show()
