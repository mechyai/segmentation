"""SLIC - Simple Linear Iterative Clustering"""

import numpy as np
from typing import Union, Tuple


def generate_SLIC_primitives(
    image: np.ndarray,
    k_clusters: int,
    initialization_neighborhood: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Peform Simple Linear Iterative Clustering (SLIC) algorithm on greyscale or RGB image

    :param image: (H, W, 3) color or (H, W) greyscale image
    :param k_clusters: Number of SLIC clusters to generate within the image
    :param initialization_neighborhood: Square area about initially sampled cluster centers, to adjust them to the lowest gradient position
    :return: (H, W) array of labels, (H, W) array of distances to the pixels corresponding cluster center, & (K, 2) final cluster centers
    """
    # Get initial sampling of cluster centers by grid


if __name__ ==  "__main__":
    pass