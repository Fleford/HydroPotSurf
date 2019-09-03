import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import winsound
import time
from numba import njit, prange
from planar_cosine_waves import generate_cosine_array, generate_sine_array, diagonal_counter
np.set_printoptions(linewidth=300)

# Version 2.2 changes how perturbation arrays are applied


@njit()
def shift_matrix_sum(matrix):
    h = matrix.shape[0]
    w = matrix.shape[1]
    result_matrix = np.zeros_like(matrix)

    # Pad the matrix with zeros
    matrix_padded = np.zeros((h + 2, w + 2))
    for row in prange(h):
        for col in prange(w):
            matrix_padded[row + 1, col + 1] = matrix[row, col]

    # Fill result matrix
    for row in prange(h):
        for col in prange(w):
            result_matrix[row, col] = matrix_padded[1 + row + 1, 1 + col] + \
                                      matrix_padded[1 + row - 1, 1 + col] + \
                                      matrix_padded[1 + row, 1 + col + 1] + \
                                      matrix_padded[1 + row, 1 + col - 1]

    return result_matrix


@njit()
def laplace_smooth(h_matrix, k_matrix):
    # Performs one iteration of the Laplace smoothing operation
    k_matrix = np.absolute(k_matrix)
    hk_matrix = h_matrix * k_matrix
    # # When dividing by zero, a zero is returned
    # new_hk_matrix = np.divide(shift_matrix_sum(hk_matrix), shift_matrix_sum(k_matrix),
    #                           out=np.zeros_like(shift_matrix_sum(hk_matrix)), where=shift_matrix_sum(k_matrix) != 0)
    hk_matrix_sum = shift_matrix_sum(hk_matrix)
    k_matrix_sum = shift_matrix_sum(k_matrix)
    new_hk_matrix = np.zeros_like(k_matrix_sum)
    for row in prange(k_matrix_sum.shape[0]):
        for col in prange(k_matrix_sum.shape[1]):
            if not k_matrix_sum[row, col] == 0:
                new_hk_matrix[row, col] = hk_matrix_sum[row, col] / k_matrix_sum[row, col]
            else:
                new_hk_matrix[row, col] = 0
    # new_hk_matrix = hk_matrix_sum / k_matrix_sum

    # # # Zero out regions with zero hk
    # new_hk_matrix = np.divide((new_hk_matrix * k_matrix), k_matrix,
    #                           out=np.zeros_like(new_hk_matrix * hk_matrix), where=k_matrix != 0)
    return new_hk_matrix


# w/o njit: 32secs
@njit()
def laplace_smooth_iter(h_matrix, k_matrix, convergence_threshold=0.001):
    # Performs a number of iterations of Laplace smoothing
    # h_matrix = np.ones_like(h_matrix) * h_field.max()

    # Make mask and copy of initial h_matrix for constant heads (where k is negative)
    # one_at_neg_k = np.ma.masked_less(k_matrix, 0).mask * 1
    one_at_neg_k = np.zeros_like(k_matrix)
    for row in prange(k_matrix.shape[0]):
        for col in prange(k_matrix.shape[1]):
            if k_matrix[row, col] < 0:
                one_at_neg_k[row, col] = 1

    # zero_at_neg_k = np.ma.masked_greater_equal(k_matrix, 0).mask * 1
    zero_at_neg_k = np.ones_like(k_matrix)
    for row in prange(k_matrix.shape[0]):
        for col in prange(k_matrix.shape[1]):
            if k_matrix[row, col] < 0:
                zero_at_neg_k[row, col] = 0

    initial_h_matrix = h_matrix.copy()

    # Sanitize k_matrix from negatives
    k_matrix_abs = np.absolute(k_matrix)

    while True:
        # Perform smoothing
        new_h_matrix = laplace_smooth(h_matrix, k_matrix_abs)
        # Apply constant heads (where k is negative)
        new_h_matrix = zero_at_neg_k * new_h_matrix + one_at_neg_k * initial_h_matrix
        # Zero out const heads on zero k
        # new_h_matrix = np.divide((new_h_matrix * k_matrix_abs), k_matrix_abs,
        #                          out=np.zeros_like(new_h_matrix * k_matrix_abs), where=k_matrix_abs != 0)
        new_h_matrix_times_k_matrix_abs = new_h_matrix * k_matrix_abs
        for row in prange(k_matrix_abs.shape[0]):
            for col in prange(k_matrix_abs.shape[1]):
                if not k_matrix_abs[row, col] == 0:
                    new_h_matrix[row, col] = new_h_matrix_times_k_matrix_abs[row, col] / k_matrix_abs[row, col]
                else:
                    new_h_matrix[row, col] = 0
        # new_h_matrix = new_h_matrix_times_k_matrix_abs / k_matrix_abs

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


# Key component 1
def calculate_boundary_values(obs_matrix, k_cnst_obs, k_cnst_bnd, convergence_threshold=0.1):
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

        # # Stop if the change switches direction
        # print(max_diff)
        # if max_diff > prev_max_diff:
        #     break

        # Stop if the change is too small
        print(max_diff)
        if max_diff < convergence_threshold:
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


def calculate_total_head_error(input_wavy_array, input_scale_factor, input_h_matrix, input_k_matrix,
                               input_obs_matrix):
    # Note that the input_k_matrix is the k matrix before application of the cosine array
    # Build index list for all observed heads
    obs_indexes = np.argwhere(input_obs_matrix)

    # Generate perturbation array
    perturbation_array = 2 ** (input_wavy_array * input_scale_factor)

    # Apply perturbation to k_field
    trial_k_matrix = input_k_matrix * perturbation_array

    # Calculate new h_field
    cal_h_matrix = laplace_smooth_iter(input_h_matrix, trial_k_matrix)

    # Calculate total head error
    total_head_error = 0
    for obs_cood in obs_indexes:
        # Calculate head error
        head_error = cal_h_matrix[obs_cood[0], obs_cood[1]] - input_obs_matrix[obs_cood[0], obs_cood[1]]
        # Update total error
        total_head_error += abs(head_error)

    # Report total head error
    return total_head_error


def calculate_new_k_field_cosine_plane(h_matrix, k_matrix, obs_matrix):
    # Calculates a slightly better k_matrix using 2d DFT planes
    # h_matrix = The h_field generated by the provided k_matrix
    # Obs_matrix = zero at all values other than observation values

    # Initialize flags and counters
    k_matrix_new = k_matrix.copy()
    plane_cnt = 0
    perturbation_array = np.ones_like(k_matrix_new)
    null_cosine_array = np.zeros_like(k_matrix_new)

    base_total_head_error = calculate_total_head_error(null_cosine_array, 0, h_matrix, k_matrix, obs_matrix)

    # Start of loop
    while True:
        # Increment counter
        plane_cnt += 1

        # Generate cosine array
        m, n = diagonal_counter(plane_cnt)
        cosine_array = generate_cosine_array(k_matrix, m, n)

        # Apply cosine plane to perturbation array
        test_scale_factors = [-0.01, 0.01]
        for scale_factor in test_scale_factors:
            total_head_error = calculate_total_head_error(cosine_array, scale_factor, h_matrix, k_matrix, obs_matrix)
            if total_head_error < base_total_head_error:
                perturbation_array = perturbation_array * (2 ** (cosine_array * scale_factor))
                # print(scale_factor)
                # print(total_head_error)
                break
        # Break when frequency limit is reached
        if m + n == 10:
            break

    # Apply perturbation array to k_field
    k_matrix_new = k_matrix * perturbation_array

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

    # obs_mask
    obs_mask = obs_field.copy()
    obs_mask[obs_mask >= 1] = 1
    # print(obs_mask)

    # k_field_cnst_adj
    # Cells adjacent to obs cells become constant (set to -1)
    k_field_cnst_adj = k_field_cnst_obs.copy()
    k_field_cnst_adj_sign, k_field_cnst_adj_mag = split_into_sign_and_magnitude(k_field_cnst_adj)
    k_field_cnst_adj_sign[k_field_cnst_adj_sign != -1] = 0
    k_field_cnst_adj_sign = shift_matrix_sum(k_field_cnst_adj_sign)
    k_field_cnst_adj_sign[k_field_cnst_adj_sign < 0] = -1
    k_field_cnst_adj_sign[k_field_cnst_adj_sign == 0] = 1
    k_field_cnst_adj = k_field_cnst_adj_sign * k_field_cnst_adj_mag

    return k_field_cnst_bnd, k_field_cnst_obs, obs_field, obs_mask, k_field_cnst_adj


# Load in matrices

if __name__ == "__main__":
    # initial_input = np.loadtxt("InputFolder/initial_input.txt")
    initial_input = np.loadtxt("InputFolder/output_array_1000.out")
    print(initial_input.shape)

    k_field, k_field_const_obs, obs_field, obs_mask, k_field_const_adj = input_matrix_to_parameter_matrices(initial_input)

    # Calculate bnd heads and initial h field
    print("Calculating boundary heads")
    h_field = calculate_boundary_values(obs_field, k_field_const_obs, k_field)

    print()
    print("Calculating k field...")
    start_time = time.time()
    for run in range(16):
        k_field = calculate_new_k_field_cosine_plane(h_field, k_field, obs_field)
        h_field = laplace_smooth_iter(h_field, k_field)
    end_time = time.time()
    print()
    print("Seconds elapsed: " + str(end_time - start_time))
    plt.matshow(k_field)
    plt.show()
    plt.matshow(h_field)
    plt.show()



    # h_field = h_field_best
    # k_field = k_field2_new

    # print()
    # print("Best error: " + str(error_best))

    # # run gw_model with old and new k_field
    # h_with_old_k = laplace_smooth_iter(h_field, k_field)
    # h_with_new_k = laplace_smooth_iter(h_field, k_field2_new)
    # # plt.matshow(h_with_new_k - h_with_old_k)
    # plt.matshow(k_field2_new)
    # new_h_field = h_with_new_k
    # plt.show()

    # # Iteratively adjust k
    # new_h_field, new_k_field = iteratively_adjust_k(h_field, k_field, obs_field)
    # # print("hup!")
    # # print(new_h_field)
    # # print(new_k_field)

    # new_h_field = h_field

    # Calculate error field
    # error_field = obs_mask * (new_h_field - obs_field)
    # error_field = laplace_smooth_iter(error_field, k_field_const_obs)
    # error_field = error_field / (new_h_field.max() - new_h_field.min())
    # plt.matshow(error_field)
    # plt.show()

    # Display current results
    levels = np.arange(np.amin(h_field), np.amax(h_field), 0.2)
    # plt.matshow(h_field)
    # plt.matshow(new_h_field)
    # plt.matshow(new_h_field - obs_field)

    # Plot 3d Surface
    fig = plt.figure()
    ax = Axes3D(fig, azim=-128.0, elev=43.0)

    h_field[h_field == 0] = np.nan
    Z = h_field
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    ax.plot_surface(X, Y, Z)
    plt.show()

    # h_field[h_field == 0] = np.nan
    plt.matshow(k_field)
    plt.contour(h_field, levels=levels)

    # plt.matshow(new_k_field)
    # # empty = np.zeros_like(new_h_field)
    # # plt.matshow(empty, cmap="gray")
    # # new_h_field[new_h_field == 0] = np.nan
    # plt.contour(new_h_field, levels=levels)
    plt.show()
