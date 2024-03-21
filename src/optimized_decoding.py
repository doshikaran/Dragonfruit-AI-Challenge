import numpy as np

def decode_rle_optimized(rle_encoded, shape):
    """
    Decode a run-length encoded (RLE) array to its original array representation,
    optimized for efficiency.

    This function reconstructs the original array from its RLE format, where the
    encoding consists of alternating values and their frequencies.

    Parameters:
    - rle_encoded (numpy.ndarray): The RLE encoded data as a 1D array.
    - shape (tuple): The shape of the original array to be reconstructed.

    Returns:
    - numpy.ndarray: The decoded 2D array of the specified shape.
    """
    decoded = np.zeros(shape, dtype=np.uint8)
    ends = np.cumsum(rle_encoded[1::2])  
    starts = ends - rle_encoded[1::2]  
    values = rle_encoded[::2]
    for start, end, value in zip(starts, ends, values):
        decoded.ravel()[start:end] = value
    
    return decoded

def decode_sparse_matrix_optimized(sparse_matrix, shape):
    """
    Decode a sparse matrix representation to its original 2D array form,
    optimized for efficiency.

    A sparse matrix, represented as a list of (row, column, value) tuples for
    non-zero elements, is converted back to the original dense 2D array format.

    Parameters:
    - sparse_matrix (numpy.ndarray): A 2D array of sparse matrix entries.
    - shape (tuple): The shape of the original 2D array to be reconstructed.

    Returns:
    - numpy.ndarray: The decoded 2D array of the specified shape.
    """
    decoded = np.zeros(shape, dtype=np.uint8)
    rows, cols, values = sparse_matrix.T
    decoded[rows, cols] = values
    return decoded
