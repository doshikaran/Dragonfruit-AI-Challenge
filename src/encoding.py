import numpy as np
import json

WIDTH, HEIGHT = 1000, 1000

def encode_rle(blob):
    """
    Encode a 2D array (blob) into a run-length encoding (RLE) array.

    RLE compresses the blob by replacing sequences of identical pixels (runs) with just
    one occurrence of the pixel followed by its run length. This function encodes the blob
    into an RLE array using this method, which can significantly reduce the size for arrays
    with large runs of identical values.

    Parameters:
    - blob (numpy.ndarray): A 2D numpy array representing the image or a section of an image
                            to be encoded. The blob's pixel values should be of a type that can
                            be compared for equality.

    Returns:
    - numpy.ndarray: A 1D array of type np.uint16, where each pair of elements represents a 
                     pixel value and its run length in the original blob. The array length is
                     variable depending on the content of the blob.

    Note:
    - The maximum run length that can be encoded is 65535 due to the use of np.uint16. Runs
      longer than this are split into multiple entries.
    """
    rle = []
    prev_pixel = None
    run_length = 0
    
    for pixel in np.ravel(blob):
        if pixel != prev_pixel or run_length == 65535:
            if run_length > 0:
                rle.extend([prev_pixel, run_length])
            prev_pixel = pixel
            run_length = 1
        else:
            run_length += 1
    
    if run_length > 0:
        rle.extend([prev_pixel, run_length])
    
    return np.array(rle, dtype=np.uint16)

def encode_sparse_matrix(dye_distribution):
    """
    Encode a 2D array representing dye distribution into a sparse matrix representation.

    A sparse matrix is a compact way of storing and working with matrices that have a large
    number of zero values. This function converts a 2D numpy array into a sparse matrix
    format, which is essentially a list of (row, column, value) tuples for non-zero elements.

    Parameters:
    - dye_distribution (numpy.ndarray): A 2D numpy array where non-zero values represent
                                        the presence of dye and zeros represent its absence.

    Returns:
    - numpy.ndarray: A 2D numpy array where each row corresponds to a non-zero element in
                     the original array. Each row contains three values: the row index,
                     the column index, and the value of the element (in this case, 1, 
                     indicating the presence of dye).

    Note:
    - This implementation assumes binary data in `dye_distribution`, and thus the value
      in the sparse matrix is always set to 1. For non-binary data, modifications would
      be necessary to capture the actual values from `dye_distribution`.
    """
    rows, cols = np.where(dye_distribution)
    data = np.ones_like(rows, dtype=np.uint8)
    sparse_matrix = np.column_stack((rows, cols, data))
    
    return sparse_matrix
