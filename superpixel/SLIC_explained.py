"""SLIC - Simple Linear Iterative Clustering"""

import logging
from typing import List, Tuple

import numpy as np
from scipy import spatial, interpolate
from skimage import color


def generate_SLIC_primitives(
    image: np.ndarray,
    k_clusters: int,
    iterations: int = 10,
    m: int = 10,
    initialization_neighborhood: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Simple Linear Iterative Clustering (SLIC) algorithm on greyscale or RGB image

    :param image: (H, W, 3) sRGB-color or (H, W) greyscale image
    :param k_clusters: Number of SLIC clusters to generate within the image
    :param iterations: Number of cycles through each cluster center to adjust its cluster contents
    :param m: TODO
    :param initialization_neighborhood: Square area about initially sampled cluster centers, to adjust them to the lowest gradient position
        This parameter defines the length of the square, in pixels, and must be an odd number
    :return: (H, W) array of labels, (H, W) array of distances to the pixels corresponding cluster center, & (K, 2) final cluster centers
    """
    # Input validation
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Image must be 2D greyscale or 3D sRGB, but got <{len(image.shape)}D>")

    if len(image.shape) == 3 and image.shape[-1] != 3:
        raise ValueError(f"Image must be 2D greyscale or 3D sRGB, but got <{image.shape[-1]}> channels")

    if initialization_neighborhood % 2 == 0:
        raise ValueError(f"Initialization neighborhood must be an odd number, but got <{initialization_neighborhood}>")

    if k_clusters < 1:
        raise ValueError(f"Number of clusters must be greater than 0, but got <{k_clusters}>")

    if iterations < 1:
        raise ValueError(f"Number of iterations must be greater than 0, but got <{iterations}>")

    ###############
    # Task metadata
    ###############
    # Image properties
    height = image.shape[0]
    width = image.shape[1]
    image_shape = image.shape[:2]

    # ~ Number of image pixels
    N = height * width

    # ~ Pixel area of square clusters
    S = N / k_clusters

    # ~ Cluster dimensions
    cluster_length = np.sqrt(S)

    # Convert RGB image to LAB colorspace
    image_LAB = color.rgb2lab(image)

    ######################################################
    # Get initial cluster center locations via grid points
    ######################################################
    # Get dimensions by the closest factor pair to the image ratio
    factor_pairs = np.array(get_factor_pairs(k_clusters, unique=False))
    factor_pairs_and_image_ratio_difference = np.abs(factor_pairs[:, 0] / factor_pairs[:, 1] - height / width)
    factor_pair = factor_pairs[np.argmin(factor_pairs_and_image_ratio_difference)]

    # Number of clusters per row and column
    row_wise_cluster_count = factor_pair[0]
    column_wise_cluster_count = factor_pair[1]

    # Select centered grid points for cluster centers
    cluster_rows = np.linspace(start=0, stop=height, num=row_wise_cluster_count + 2, endpoint=True, dtype="int16")[1:-1]
    cluster_columns = np.linspace(start=0, stop=width, num=column_wise_cluster_count + 2, endpoint=True, dtype="int16")[1:-1]

    cluster_centers_columns, cluster_centers_rows = np.meshgrid(cluster_columns, cluster_rows, sparse=False)

    # Create list of cluster center coordinates
    cluster_center_coordinates_bitmap = np.concatenate(
        (np.expand_dims(cluster_centers_rows, axis=-1), np.expand_dims(cluster_centers_columns, axis=-1)),
        axis=-1,
    ).reshape(-1, 2)

    # Create sparse matrix with shape of input image with depth for (label + LAB colorspace) to track cluster centers spatially & in color
    # We will populate this with the cluster center labels and the LAB colorspace values in the loop below
    cluster_center_coordinates_sparse_matrix = np.zeros(shape=image_shape + (4,))  # H x W x 4 (label + LAB colorspace) matrix

    ##################################################
    # Adjust cluster centers to local minimum gradient
    ##################################################
    # Get image gradient
    image_gradient_y, image_gradient_x, _ = np.gradient(image)
    # Get total magnitude of the x & y directional gradients
    gradient_magnitude = np.linalg.norm(image_gradient_y, axis=-1) + np.linalg.norm(image_gradient_x, axis=-1)

    for index, current_cluster_center_coordinate in enumerate(cluster_center_coordinates_bitmap):
        # Get current cluster data
        row, column = current_cluster_center_coordinate

        LAB_color = image_LAB[row, column]
        cluster_label = np.array([index + 1])  # Start labeling at 1

        # Get neighborhood slice about center point
        neighborhood_window = get_window_about_center_pixel_coordinate(
            image_shape=image_shape, center_pixel_coordinate=(row, column), window_size=initialization_neighborhood,
        )

        # Get gradient values within neighborhood, find the minimum(s)
        neighborhood_gradient = gradient_magnitude[neighborhood_window]
        neighborhood_center_index = np.array([[int(neighborhood_gradient.shape[0] // 2), int(neighborhood_gradient.shape[1]) // 2]])

        minimum_gradient_relative_indices = np.argwhere(neighborhood_gradient == np.min(neighborhood_gradient))

        # If multiple minimums found, select the coordinate nearest to the original center (i.e. adjust the center the least distance)
        if len(minimum_gradient_relative_indices) > 1:
            minimum_gradient_distances_from_center = spatial.distance.cdist(
                minimum_gradient_relative_indices, neighborhood_center_index
            ).reshape(-1)
            minimum_gradient_index = minimum_gradient_relative_indices[np.argmin(minimum_gradient_distances_from_center)]
        else:
            minimum_gradient_index = minimum_gradient_relative_indices[0]

        # Get new cluster center coordinate relative to original center
        minimum_gradient_relative_index = minimum_gradient_index - neighborhood_center_index

        adjusted_cluster_center_coordinate = current_cluster_center_coordinate + minimum_gradient_relative_index
        new_row, new_column = adjusted_cluster_center_coordinate[0]

        # Adjust current cluster data, if moved from original location
        if not np.array_equal(current_cluster_center_coordinate, adjusted_cluster_center_coordinate[0]):
            # Update cluster center location bitmap
            cluster_center_coordinates_bitmap[index] = adjusted_cluster_center_coordinate
            # Erase old cluster location data from sparse matrix
            cluster_center_coordinates_sparse_matrix[new_row, new_column, :] = np.zeros(shape=(4,))

        # Populate the final cluster center coordinates sparse matrix with the cluster center label and the LAB colorspace values
        cluster_center_coordinates_sparse_matrix[new_row, new_column, :] = np.concatenate((cluster_label, LAB_color), axis=-1)

    ###########################################
    # Create Voronoi diagram of cluster centers
    ###########################################
    # Create labels and distance matrix to track cluster assignments and distances
    # Draw Voronoi diagram https://stackoverflow.com/a/71734591
    grid_x, grid_y = np.mgrid[0:height, 0:width]
    labels_matrix = interpolate.griddata(
        points=cluster_center_coordinates_bitmap, values=np.arange(1, k_clusters + 1), xi=(grid_x, grid_y), method="nearest"
    )

    distances_matrix = np.full(shape=image_shape, fill_value=np.inf, dtype="float32")

    #############################
    # Iteratively adjust clusters
    #############################
    # Create pixel coordinate matrix for easy windowing and indexing - same shape as image
    row_spacing = np.arange(stop=height, dtype="int16")
    column_spacing = np.arange(stop=width, dtype="int16")

    pixel_columns, pixel_rows = np.meshgrid(column_spacing, row_spacing, sparse=False)
    pixel_coordinates_matrix = np.concatenate((np.expand_dims(pixel_rows, axis=-1), np.expand_dims(pixel_columns, axis=-1)), axis=-1)

    # For each iteration
    for iteration in range(iterations):
        logging.info(f"\tRunning SLIC iteration {iteration + 1} of {iterations}...")

        # For cluster center
        for index, current_cluster_center_coordinate in enumerate(cluster_center_coordinates_bitmap):
            logging.info(f"\t\tAt cluster center {index + 1} of {k_clusters} at <{current_cluster_center_coordinate}>...")

            # Get 2S x 2S window of pixels about cluster center
            cluster_window = get_window_about_center_pixel_coordinate(
                image_shape=image_shape, center_pixel_coordinate=current_cluster_center_coordinate, window_size=2 * int(cluster_length),
            )

            # Get the slice coordinates relative to the entire image
            image_coordinates_slice = pixel_coordinates_matrix[cluster_window]
            # Get the slice coordinates relative to the cluster window, normalize by the top-left corner
            window_coordinates_slice = image_coordinates_slice - image_coordinates_slice[0, 0]

            # Get matrix of pixel distance - for each pixel in window to cluster center
            pixel_distances_from_center = spatial.distance.cdist(
                image_coordinates_slice.reshape(-1, 2), current_cluster_center_coordinate.reshape(1, -1)
            ).reshape(image_coordinates_slice.shape[:2])

            # Get matrix of color distances - for each pixel in window to cluster center
            cluster_center_color_vector = (
                cluster_center_coordinates_sparse_matrix[current_cluster_center_coordinate[0], current_cluster_center_coordinate[1], 1:]
            )

            LAB_colors_in_window = image_LAB[cluster_window]

            color_distances_from_center = spatial.distance.cdist(
                LAB_colors_in_window.reshape(-1, 3), cluster_center_color_vector.reshape(1, -1)
            ).reshape(image_coordinates_slice.shape[:2])

            # For each pixel in window - adjust label & distance matrix if new distance is less than current distance
            for window_coordinate, image_coordinate in zip(window_coordinates_slice.reshape(-1, 2), image_coordinates_slice.reshape(-1, 2)):
                # Skip cluster center pixel
                if np.array_equal(image_coordinate, current_cluster_center_coordinate):
                    continue

                # Compute distance metric for 5D space: sqrt(pixel_distance^2 + (m/S)^2 * color_distance^2)
                pixel_distance = pixel_distances_from_center[window_coordinate[0], window_coordinate[1]]
                color_distance = color_distances_from_center[window_coordinate[0], window_coordinate[1]]

                distance_from_current_cluster_center = np.sqrt(pixel_distance ** 2 + (m / S) ** 2 * color_distance ** 2)

                # Update label & distance matrix if new distance is less than current distance
                current_distance = distances_matrix[window_coordinate[0], window_coordinate[1]]
                if distance_from_current_cluster_center < current_distance:
                    # Update label & distance matrix
                    labels_matrix[image_coordinate[0], image_coordinate[1]] = index + 1  # Start labeling at 1
                    distances_matrix[image_coordinate[0], image_coordinate[1]] = distance_from_current_cluster_center

    return labels_matrix, distances_matrix, cluster_center_coordinates_bitmap


def get_window_about_center_pixel_coordinate(
    image_shape: Tuple[int, int],
    center_pixel_coordinate: Tuple[int, int],
    window_size: int,
) -> Tuple[slice, slice]:
    """
    Get window slice about center pixel coordinate for a 2D image shape. Windows intersecting image boundaries are truncated

    :param image_shape: 2D image shape, (H, W)
    :param center_pixel_coordinate: Center pixel coordinate for which the window will be created, (row, column)
    :param window_size: Window size, in pixels. If not an odd number, the center pixel coordinate will be right of center of the window
    :return: Slice for the row and column axes of numpy array
    """
    image_height, image_width = image_shape
    row_center, column_center = center_pixel_coordinate

    # Get neighborhood slice about center point
    row_slice_minimum = int(max(0, row_center - window_size // 2))
    row_slice_maximum = int(min(image_height, row_center + window_size // 2 + 1))

    column_slice_minimum = int(max(0, column_center - window_size // 2))
    column_slice_maximum = int(min(image_width, column_center + window_size // 2 + 1))

    return np.s_[row_slice_minimum: row_slice_maximum], np.s_[column_slice_minimum: column_slice_maximum]


def get_factor_pairs(number: int, unique: bool = False) -> List[Tuple[int, int]]:
    """
    Find all pairs of factors of a given number

    :param number: Number to find factor pairs for
    :param unique: If True, only return unique factor pairs, only (i, j) or (j, i) will be returned, not both
    :return: Complete set, or unique, factor pairs
    """
    factor_pairs = []
    for i in range(1, int(np.sqrt(number)) + 1):
        if number % i == 0:
            factor_pairs.append((number // i, i))
            if not unique:
                factor_pairs.append((i, number // i))

    return factor_pairs


if __name__ == "__main__":
    import cv2
    from PIL import Image

    logging.basicConfig(level=logging.INFO)

    # Load image
    image_path = "/home/cje/Downloads/dog.png"
    image = Image.open(image_path)
    # TODO even with square image there are still weird 0 patches that seem to be untouched by the window
    image = image.resize((500, 500))  # TODO algorithm doesn't handle non near square images well
    image = np.array(image)


    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    logging.info("Running SLIC...")

    labels_matrix, distances_matrix, _ = generate_SLIC_primitives(
        image=image,
        k_clusters=100,
        iterations=2,
        m=10,
        initialization_neighborhood=3,
    )
