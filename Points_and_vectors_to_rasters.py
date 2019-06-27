import numpy as np

# The goal is to convert points or vectors into rasters with lines or polygons


def list_all_intersected_cells(coord_start, coord_end):
    # Function that lists all cells that intersects the line defined by two points
    # y - y1 = m(x - x1)
    dx_dy = coord_end - coord_start

    # Prep list of points
    list_of_cells = []
    if np.abs(dx_dy[0]) > np.abs(dx_dy[1]):
        m = dx_dy[1] / dx_dy[0]
        y = lambda x: m * (x - coord_start[0]) + coord_start[1]
        list_of_cells.append(tuple(coord_start))
        x_position = coord_start[0] + (dx_dy[0] / np.abs(dx_dy[0])).astype(int)
        # Add intersected points to list
        while x_position != coord_end[0]:
            x_low = x_position - 0.5
            x_high = x_position + 0.5
            y_low = y(x_low).round().astype(int)
            y_high = y(x_high).round().astype(int)
            # print(f"x_position {x_position}, x_low {x_low}, x_high {x_high}, y_low {y_low}, y_high {y_high}")
            for y_position in range(min(y_low, y_high), max(y_low, y_high) + 1):
                # print(y_position)
                list_of_cells.append((x_position, y_position))
            # print(list_of_cells)
            x_position += (dx_dy[0] / np.abs(dx_dy[0])).astype(int)
        list_of_cells.append(tuple(coord_end))
    else:
        m = dx_dy[0] / dx_dy[1]
        x = lambda y: m * (y - coord_start[1]) + coord_start[0]
        list_of_cells.append(tuple(coord_start))
        y_position = coord_start[1] + (dx_dy[1] / np.abs(dx_dy[1])).astype(int)
        # Add intersected points to list
        while y_position != coord_end[1]:
            y_low = y_position - 0.5
            y_high = y_position + 0.5
            x_low = x(y_low).round().astype(int)
            x_high = x(y_high).round().astype(int)
            for x_position in range(min(x_low, x_high), max(x_low, x_high) + 1):
                list_of_cells.append((x_position, y_position))
            y_position += (dx_dy[1] / np.abs(dx_dy[1])).astype(int)
        list_of_cells.append(tuple(coord_end))

    return list_of_cells


def point_array_list_all_intersected_cells(point_array):
    # Given multiple points, this functions draws a line between two adjacent points.
    # The function then returns a list of all intersected cells
    return None



array = np.zeros((11, 11))
start_point = np.array([0, 0])
intermediate_point = np.array([3, 3])
end_point = np.array([5, 5])

array[tuple(start_point)] = 1
array[tuple(end_point)] = 2

list = list_all_intersected_cells(start_point, end_point)

# Add points to matrix
for coord in list:
    array[coord] = 3

print(list)
print(array)

