"""SLIC - Simple Linear Iterative Clustering"""

from typing import Union, Tuple

import numpy as np
from scipy import spatial


def generate_SLIC_primitives(
    image: np.ndarray,
    k_clusters: int,
    m: int = 10,
    initialization_neighborhood: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Peform Simple Linear Iterative Clustering (SLIC) algorithm on greyscale or RGB image

    :param image: (H, W, 3) color or (H, W) greyscale image
    :param k_clusters: Number of SLIC clusters to generate within the image
    :param m: TODO
    :param initialization_neighborhood: Square area about initially sampled cluster centers, to adjust them to the lowest gradient position
    :return: (H, W) array of labels, (H, W) array of distances to the pixels corresponding cluster center, & (K, 2) final cluster centers
    """
    height = image.shape[0]
    width = image.shape[1]

    # ~ Number of image pixels
    N = height * width

    # ~ Pixel area of square clusters
    S = N / k_clusters

    # ~ Cluster dimensions
    cluster_length = np.sqrt(S)
    remainder = cluster_length - int(cluster_length)

    # Get initial cluster center locations via grid points
    # Sample evenly across image width and height, centered in the middle
    row_spacing = np.arange(start=remainder / 2, stop=height - remainder / 2, step=cluster_length, dtype="uint16")
    column_spacing = np.arange(start=remainder / 2, stop=width - remainder / 2, step=cluster_length, dtype="uint16")

    cluster_centers_columns, cluster_centers_rows = np.meshgrid(column_spacing, row_spacing, sparse=False)

    cluster_center_coordinates_bitmap = np.concatenate(
        (np.expand_dims(cluster_centers_rows, axis=-1), np.expand_dims(cluster_centers_columns, axis=-1)),
        axis=-1,
    )
    cluster_center_coordinates_sparse_matrix = np.zeros_like(image[:, :, 0], dtype="uint16")

    # Find starting cluster centers at minimum gradient within some neighborhood around the grid points
    image_gradient_y, image_gradient_x, _ = np.gradient(image)

    # Get total magnitude of the x & y directional gradients
    gradient_magnitude = np.linalg.norm(image_gradient_y, axis=-1) + np.linalg.norm(image_gradient_x, axis=-1)

    for row_index in range(len(row_spacing)):
        for column_index in range(len(column_spacing)):
            # Get neighborhood slice about center point
            current_cluster_location = cluster_center_coordinates_bitmap[row_index, column_index]
            row, column = current_cluster_location

            row_slice_minimum = int(max(0, row - initialization_neighborhood // 2))
            row_slice_maximum = int(min(image.shape[0], row + initialization_neighborhood // 2 + 1))

            column_slice_minimum = int(max(0, column - initialization_neighborhood // 2))
            column_slice_maximum = int(min(image.shape[1], column + initialization_neighborhood // 2 + 1))

            # Get gradient values within neighborhood, find the minimum(s)
            neighborhood_gradient = gradient_magnitude[row_slice_minimum: row_slice_maximum, column_slice_minimum: column_slice_maximum]
            neighborhood_center_index = np.array([[int(neighborhood_gradient.shape[0] // 2), int(neighborhood_gradient.shape[1]) // 2]])

            minimum_gradient_relative_indices = np.argwhere(neighborhood_gradient == np.min(neighborhood_gradient))

            # If multiple minimums found, select the coordinate nearest to the original center (i.e. adjust the center the least distance)
            if len(minimum_gradient_relative_indices) > 0:
                minimum_gradient_distances_from_center = spatial.distance.cdist(
                    minimum_gradient_relative_indices, neighborhood_center_index
                ).reshape(-1)
                minimum_gradient_index = minimum_gradient_relative_indices[np.argmin(minimum_gradient_distances_from_center)]
            else:
                minimum_gradient_index = minimum_gradient_relative_indices[0]

            minimum_gradient_relative_index = minimum_gradient_index - neighborhood_center_index

            # Update cluster center location, if moved
            if minimum_gradient_relative_index.any():
                cluster_center_coordinates_bitmap[row_index, column_index] = current_cluster_location + minimum_gradient_relative_index

    pass


if __name__ == "__main__":
    _ = generate_SLIC_primitives(
        image=np.ones((100, 50, 3)),
        k_clusters=100,
    )
