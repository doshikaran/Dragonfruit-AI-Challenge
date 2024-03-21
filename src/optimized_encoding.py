import numpy as np

def encode_rle_optimized(blob):
    """
    Efficiently encodes a 2D array (blob) into a run-length encoding (RLE) format.

    This optimized version traverses the flattened array of the blob, aggregating consecutive
    identical elements into [value, run length] pairs, thereby compressing the data.

    Parameters:
    - blob (numpy.ndarray): The 2D array to be encoded, where the blob's content is represented.

    Returns:
    - numpy.ndarray: A 1D array of alternating values and their run lengths, encoded in RLE format,
                     with a dtype of np.uint16 to accommodate potentially large run lengths.

    Note:
    - This function assumes that the input blob is a binary array, though it may work with
      any array where elements can be compared for equality.
    """
    rle = []
    flat_blob = np.ravel(blob)
    prev_pixel = flat_blob[0]
    run_length = 1

    for pixel in flat_blob[1:]:
        if pixel == prev_pixel:
            run_length += 1
        else:
            rle.extend([prev_pixel, run_length])
            prev_pixel = pixel
            run_length = 1
    rle.extend([prev_pixel, run_length]) 
    
    return np.array(rle, dtype=np.uint16)


def encode_sparse_matrix_optimized(dye_distribution):
    """
    Converts a 2D array representing dye distribution into an optimized sparse matrix format.

    This function identifies non-zero elements in the dye distribution and represents each
    by its row and column indices, alongside the value of the element, effectively compressing
    the data into a more storage-efficient format.

    Parameters:
    - dye_distribution (numpy.ndarray): A 2D array where non-zero values represent the presence
                                        of dye, intended for encoding into sparse matrix format.

    Returns:
    - numpy.ndarray: An array of shape (n, 3), where 'n' is the number of non-zero elements in
                     the original array. Each row contains the row index, column index, and value
                     (always 1 in this binary context) of these non-zero elements.

    Note:
    - The output is particularly useful for storing and processing data with a high proportion
      of zeros, as is common in certain types of imaging data.
    """
    rows, cols = np.where(dye_distribution)
    data = np.ones_like(rows, dtype=np.uint8) 
    sparse_matrix = np.column_stack((rows, cols, data))
    return sparse_matrix
