import numpy as np

WIDTH, HEIGHT = 1000, 1000

def generate_blob(radius_range=(100, 400)):
    """
    Generates a circular blob within a specified radius range on a 2D grid.

    This function creates a binary array of shape (HEIGHT, WIDTH), representing an image
    where a circular blob is marked by True (1) values, and the background is False (0).
    The position of the blob is randomly determined, as is its radius, within the provided range.

    Parameters:
    - radius_range (tuple of int, optional): A tuple specifying the minimum and maximum
      possible radii for the blob. Defaults to (100, 400).

    Returns:
    - numpy.ndarray: A 2D binary numpy array of shape (HEIGHT, WIDTH) with a single blob
                     generated according to the specified parameters.

    Note:
    - The generated blob is guaranteed not to touch the edges of the image, adhering to the
      specified radius constraints.
    """
    radius = np.random.randint(radius_range[0], radius_range[1])
    x0, y0 = np.random.randint(radius, WIDTH - radius), np.random.randint(radius, HEIGHT - radius)
    Y, X = np.indices((HEIGHT, WIDTH))
    dist_from_center = np.sqrt((X - x0)**2 + (Y - y0)**2)
    blob = dist_from_center <= radius
    return blob

def generate_dye_distribution(blob, dye_ratio=0.1):
    """
    Generates a random dye distribution for a given blob within a 2D space.

    The dye distribution is binary, where True (1) indicates the presence of dye, and False
    (0) indicates its absence. The function applies a higher probability for dye presence
    inside the blob than outside, simulating a common pattern in biological imaging where
    a region of interest reacts differently to staining or dyeing processes.

    Parameters:
    - blob (numpy.ndarray): A binary 2D numpy array representing an area with a blob (marked by 1)
                            against a background (marked by 0).
    - dye_ratio (float, optional): The base probability of dye presence outside the blob.
                                   Inside the blob, the probability is doubled. Defaults to 0.1.

    Returns:
    - numpy.ndarray: A 2D binary numpy array of the same shape as `blob`, indicating the dye
                     distribution, with True for dyed pixels and False for undyed pixels.

    Note:
    - The function generates dye distribution both inside and outside the blob but ensures that
      the probability of dye presence is higher within the blob to mimic selective staining.
    """
    dye_inside = np.random.rand(*blob.shape) < dye_ratio * 2
    dye_inside[~blob] = False 
    dye_outside = np.random.rand(*blob.shape) < dye_ratio
    dye_outside[blob] = False  
    dye_distribution = dye_inside | dye_outside
    return dye_distribution
