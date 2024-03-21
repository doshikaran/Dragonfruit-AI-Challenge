# 2. Generating Simulated Images
import numpy as np
import json

WIDTH, HEIGHT = 1000, 1000

# Function to generate a random blob shape
def generate_blob(radius_range=(100, 400)):
    """
    Generate a random circular blob within a predefined 2D space.

    This function creates a binary 2D numpy array representing an image of WIDTH x HEIGHT,
    where a circular blob (region of interest) is set to True (or 1), and the rest of the
    space is set to False (or 0). The position and size of the blob are randomly determined,
    constrained by the specified radius range.

    Parameters:
    - radius_range (tuple of int, optional): A tuple specifying the minimum and maximum
      radius of the blob. Defaults to (100, 400).

    Returns:
    - numpy.ndarray: A 2D binary numpy array of shape (HEIGHT, WIDTH), where the blob's
      area is represented by True (1) values, and the rest is False (0).

    Note:
    - The blob is ensured to be fully contained within the boundaries of the generated image.
    """
    radius = np.random.randint(radius_range[0], radius_range[1])
    x0 = np.random.randint(radius, WIDTH - radius)
    y0 = np.random.randint(radius, HEIGHT - radius)
    Y, X = np.ogrid[:HEIGHT, :WIDTH]
    dist_from_center = np.sqrt((X - x0)**2 + (Y - y0)**2)
    blob = dist_from_center <= radius
    return blob

# Function to generate a dye distribution
def generate_dye_distribution(blob, dye_ratio=0.1):
    """
    Generate a random dye distribution over a 2D space with a specified blob.

    This function simulates the application of a dye to both the blob (region of interest)
    and its surrounding area, with different probabilities for the inside and outside of
    the blob. The resulting distribution is represented as a binary 2D numpy array.

    Parameters:
    - blob (numpy.ndarray): A binary 2D numpy array where the blob's area is marked by True
                            (1) values, and the rest is False (0). This array defines the
                            spatial context for dye application.
    - dye_ratio (float, optional): The base probability of a pixel being dyed outside of
                                   the blob. The probability inside the blob is twice this
                                   value. Defaults to 0.1.

    Returns:
    - numpy.ndarray: A 2D binary numpy array of the same shape as `blob`, representing the
                     dye distribution. True (1) indicates the presence of dye, while False
                     (0) indicates its absence.

    Note:
    - The dye distribution inside the blob is twice as likely to occur as it is outside,
      reflecting a common scenario in imaging where a region of interest reacts differently
      to a dye or contrast agent than its surroundings.
    """
    dye_inside = np.random.rand(*blob.shape) < dye_ratio * 2
    dye_inside[~blob] = False
    dye_outside = np.random.rand(*blob.shape) < dye_ratio
    dye_outside[blob] = False
    dye_distribution = dye_inside | dye_outside
    return dye_distribution



