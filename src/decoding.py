import numpy as np

def decode_rle(rle_encoded, shape):
    """
    Decode a run-length encoded (RLE) array back to its original 2D array representation.

    RLE is a form of lossless data compression where sequences of data are stored as a 
    single data value and count, rather than as the original run. This function reconstructs
    the original array from RLE encoded data.

    Parameters:
    - rle_encoded (list or numpy.ndarray): The RLE encoded data as a sequence where pairs 
      of [value, length] occur, representing a value and its run length in the original array.
    - shape (tuple): The shape (dimensions) of the original 2D array to be reconstructed.

    Returns:
    - numpy.ndarray: A 2D array of shape `shape`, with the original data reconstructed from 
      the RLE encoding.
    """
    decoded = np.zeros(shape, dtype=np.uint8)
    current_index = 0
    for value, length in zip(rle_encoded[::2], rle_encoded[1::2]):
        decoded.ravel()[current_index:current_index+length] = value
        current_index += length
    return decoded

def decode_sparse_matrix(sparse_matrix, shape):
    """
    Decode a sparse matrix representation back to its original 2D array representation.

    A sparse matrix is represented as a list of tuples, each containing the (row, column) 
    index of a non-zero element and its value. This function reconstructs the original 2D 
    array from this sparse representation.

    Parameters:
    - sparse_matrix (list of tuples): Each tuple contains (row_index, column_index, value)
      for non-zero elements in the original 2D array.
    - shape (tuple): The shape (dimensions) of the original 2D array to be reconstructed.

    Returns:
    - numpy.ndarray: A 2D array of shape `shape`, with the original data reconstructed from
      the sparse matrix representation.
    """
    decoded = np.zeros(shape, dtype=np.uint8)
    for row, col, value in sparse_matrix:
        decoded[row, col] = value
    return decoded
