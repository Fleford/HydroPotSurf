import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import winsound
import time
from numba import njit, prange
from planar_cosine_waves import generate_cosine_array, diagonal_counter

# Version 2 incorporates uses DFT arrays to adjust the k_matrix


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


# def shift_matrix_sum(matrix):
#     # Each 2d element becomes the sum of all its adjacent elements
#     # Empty cells are padded with zero
#     return shift_matrix(matrix, "up") + shift_matrix(matrix, "down")\
#                + shift_matrix(matrix, "left") + shift_matrix(matrix, "right")

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


def calculate_total_head_error(input_cosine_array, input_scale_factor, input_h_matrix, input_k_matrix,
                               input_obs_matrix):
    # Note that the input_k_matrix is the k matrix before application of the cosine array
    # Build index list for all observed heads
    obs_indexes = np.argwhere(input_obs_matrix)

    # Generate perturbation array
    perturbation_array = 2 ** (input_cosine_array * input_scale_factor)

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
    change_plane_flag = True

    # Start of loop
    while change_plane_flag:
        # Increment counter
        plane_cnt += 1

        # Generate cosine array
        m, n = diagonal_counter(plane_cnt)
        cosine_array = generate_cosine_array(k_matrix, m, n)

        # Prepare list of scale_factors
        test_scale_factors = [-1, -0.01, 0, 0.01, 1]
        resulting_total_head_error = []
        for scale_factor in test_scale_factors:
            total_head_error = calculate_total_head_error(cosine_array, scale_factor, h_matrix, k_matrix, obs_matrix)
            resulting_total_head_error.append(total_head_error)
        print(test_scale_factors)
        print(resulting_total_head_error)
        min_index = int(np.argmin(np.asarray(resulting_total_head_error)))

        # Initialize scale_factor_final variable
        scale_factor_final = test_scale_factors[min_index]

        # Check if optimal scale factor is within search range
        if min_index == 1 or min_index == 3:
            # Initialize variables
            sf_in = test_scale_factors[2]
            error_in = resulting_total_head_error[2]
            sf_out = test_scale_factors[int(2 * (min_index - 2) + 2)]
            error_out = resulting_total_head_error[int(2 * (min_index - 2) + 2)]

            while True:
                # Evaluate new scale_factor
                sf_middle = (sf_in + sf_out) / 2
                error_middle = calculate_total_head_error(cosine_array, sf_middle, h_matrix, k_matrix, obs_matrix)
                sf_list = [sf_in, sf_middle, sf_out]
                error_list = [error_in, error_middle, error_out]
                print(sf_list)
                print(error_list)

                # Pick out new bounding sf values and updating corresponding errors
                index_min = int(np.argmin(np.asarray(error_list)))
                index_max = int(np.argmax(np.asarray(error_list)))
                index_mid = 3 - (index_min + index_max)

                # Update final scale_factor
                scale_factor_final = sf_list[index_min]

                # Check if we're done zooming in
                if abs(sf_in - sf_out) < 0.01:
                    print("Stopped due to zoom window too small")
                    break
                if index_max == 1:
                    print("Stopped due to no change in bounds of window")
                    break

                # Otherwise, continue modifying lists
                sf_in = sf_list[index_min]
                error_in = error_list[index_min]
                sf_out = sf_list[index_mid]
                error_out = error_list[index_mid]

        elif min_index == 0 or min_index == 4:
            scale_factor_final = test_scale_factors[min_index]
            print("Stopped due to minimum found at edge of search window")

        if scale_factor_final == 0:
            print("No change recommended. Trying next plane...")
            change_plane_flag = True
        else:
            # Apply final scale factor to k field
            k_matrix_new = k_matrix * (2 ** (cosine_array * scale_factor_final))
            print("scale_factor_final")
            print(scale_factor_final)

            # End loop
            change_plane_flag = False

    return k_matrix_new


def sign_matrix_for_k_adjustment(h_matrix, obs_matrix, obs_mask):
    # A function that return a matrix with -1, 0 or 1 depending on a how the k field should be adjusted

    # Prep values
    h_matrix_center = h_matrix.copy() * obs_mask
    h_matrix_up = h_matrix.copy() * shift_matrix(obs_mask, "up")
    h_matrix_down = h_matrix.copy() * shift_matrix(obs_mask, "down")
    h_matrix_left = h_matrix.copy() * shift_matrix(obs_mask, "left")
    h_matrix_right = h_matrix.copy() * shift_matrix(obs_mask, "right")

    # Calculate h gradient signs
    h_up_grad = h_matrix_up - shift_matrix(h_matrix_center, "up")
    h_up_grad_sign = safe_divide(h_up_grad, np.absolute(h_up_grad))

    h_down_grad = h_matrix_down - shift_matrix(h_matrix_center, "down")
    h_down_grad_sign = safe_divide(h_down_grad, np.absolute(h_down_grad))

    h_left_grad = h_matrix_left - shift_matrix(h_matrix_center, "left")
    h_left_grad_sign = safe_divide(h_left_grad, np.absolute(h_left_grad))

    h_right_grad = h_matrix_right - shift_matrix(h_matrix_center, "right")
    h_right_grad_sign = safe_divide(h_right_grad, np.absolute(h_right_grad))

    # Add up all gradient signs
    h_obs_grad_sign = h_up_grad_sign + h_down_grad_sign + h_left_grad_sign + h_right_grad_sign

    # Calculate h error signs
    h_error = obs_matrix - h_matrix_center
    h_error_sign = safe_divide(h_error, np.absolute(h_error))

    # Spread h error to adjacent cells
    h_error_sign_adj = shift_matrix_sum(h_error_sign)

    # Combine h_gradient and h_error
    k_delta_sign = h_obs_grad_sign * h_error_sign_adj

    return k_delta_sign


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


# def iteratively_adjust_k(h_matrix, k_matrix, obs_matrix):
#     # Iteratively adjusts the k_field until convergence is reached
#
#     # Assume provided h_matrix is the smoothest
#     # Copy the original h_matrix
#     h_matrix_smooth = h_matrix.copy()
#
#     # Setup previous error variable
#     previous_error = np.inf
#
#     for x in range(10000):
#         # Calculate new k matrix
#         k_matrix_new = calculate_new_k_field(h_matrix, k_matrix, obs_matrix, h_matrix_smooth)
#
#         # Calculate new h matrix
#         h_matrix_new = laplace_smooth_iter(h_matrix, k_matrix_new)
#
#         # Calculate maximum error
#         h_obs_mask = obs_matrix.copy()
#         h_obs_mask[h_obs_mask != 0] = 1
#         h_error = (h_matrix_new - obs_matrix) * h_obs_mask
#         h_error_abs = np.absolute(h_error)
#         print(h_error_abs.max())
#         if h_error_abs.max() > previous_error:
#             break
#
#         # Replace current k and h matrix
#         k_matrix = k_matrix_new
#         h_matrix = h_matrix_new
#
#         # Update error
#         previous_error = h_error_abs.max()
#
#     return h_matrix, k_matrix


# Load in matrices

if __name__ == "__main__":
    # initial_input = np.loadtxt("InputFolder/initial_input.txt")
    initial_input = np.loadtxt("InputFolder/output_array_1000.out")
    print(initial_input.shape)
    start_time = time.time()
    k_field, k_field_const_obs, obs_field, obs_mask, k_field_const_adj = input_matrix_to_parameter_matrices(initial_input)

    # Calculate bnd heads and initial h field
    print("Calculating boundary heads")
    h_field = calculate_boundary_values(obs_field, k_field_const_obs, k_field)

    # # Calculate new k
    # print()
    # print("Calculating delta k matrix")
    # result = sign_matrix_for_k_adjustment(h_field, obs_field, obs_mask)
    # # print(result)

    print()
    print("Calculating k field...")
    # # print(k_field_const_obs)
    # k_field2_new = k_field.copy()
    # h_field_old = h_field.copy()
    # h_field_best = h_field.copy()
    # error_old = np.inf
    # error_best = np.inf
    # scale_value = 0.1
    # pos_count = 0
    # pos_max = 1 / scale_value
    # avg_diff = 0
    # diff = 0
    # sign_list = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    # for run in range(2**18):
    #     k_field2_new, error = calculate_new_k_field_randwalk(h_field, k_field2_new, obs_field, k_field_const_adj,
    #                                                          step_scale=scale_value)
    #     h_field = laplace_smooth_iter(h_field, k_field2_new)
    #     # print(np.max(np.absolute(h_field - h_field_old)))
    #     print(error)
    #     if error > error_old:
    #         pos_count += 1
    #
    #     if error < error_best:
    #         error_best = error
    #         k_field_best = k_field2_new
    #         h_field_best = h_field
    #
    #     if pos_count >= pos_max:
    #         pos_count = 0
    #         scale_value = scale_value / 10
    #         pos_max = pos_max * 10
    #         print("Scale value changed to " + str(scale_value))
    #         print(error)
    #
    #
    #     sign_list = np.append(sign_list, diff)
    #     sign_list = sign_list[1:]
    #     # print(np.sum(sign_list))
    #     # print(sign_list)
    #
    #     # if np.sum(sign_list) == 2:
    #     #     scale_value = scale_value / 10
    #     #     print("Scale value changed to " + str(scale_value))
    #     #     sign_list = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    #
    #     # if error > error_old:
    #     #     scale_value = scale_value / 10
    #     #     print("Scale value changed to " + str(scale_value))
    #     error_old = error
    #
    #     h_field_old = h_field
    #     # if error < 0.000001:
    #     #     break
    #     if scale_value < 0.001:
    #         break

    cnt = 0
    while True:
        cnt += 1
        k_field = calculate_new_k_field_cosine_plane(h_field, k_field, obs_field)
        if cnt % 10 == 0:
            plt.matshow(k_field)
            plt.show()

    end_time = time.time()

    # h_field = h_field_best
    # k_field = k_field2_new

    # print()
    # print("Best error: " + str(error_best))

    print()
    print("Seconds elapsed: " + str(end_time - start_time))

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

    # # Display current results
    # levels = np.arange(np.amin(h_field), np.amax(h_field), 0.2)
    # # plt.matshow(h_field)
    # # plt.matshow(new_h_field)
    # # plt.matshow(new_h_field - obs_field)

    # # Plot 3d Surface
    # fig = plt.figure()
    # ax = Axes3D(fig, azim=-128.0, elev=43.0)
    #
    # new_h_field[new_h_field == 0] = np.nan
    # Z = new_h_field
    # X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    # ax.plot_surface(X, Y, Z)
    # plt.show()
    #
    # # h_field[h_field == 0] = np.nan
    # plt.matshow(k_field)
    # plt.contour(h_field, levels=levels)
    #
    # # plt.matshow(new_k_field)
    # # # empty = np.zeros_like(new_h_field)
    # # # plt.matshow(empty, cmap="gray")
    # # # new_h_field[new_h_field == 0] = np.nan
    # # plt.contour(new_h_field, levels=levels)
    # plt.show()
