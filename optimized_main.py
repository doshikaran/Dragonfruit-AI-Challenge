import os
import json
from src.optimized_encoding import encode_rle_optimized, encode_sparse_matrix_optimized
from src.optimized_decoding import decode_rle_optimized, decode_sparse_matrix_optimized
from src.optimized_generation import generate_blob, generate_dye_distribution
from src.optimized_visualization import visualize_image_optimized
from src.optimized_cancerDetection import has_cancer_dye_optimized, has_cancer_microscope_optimized

def main(data_dir='data', width=1000, height=1000):
    os.makedirs(data_dir, exist_ok=True)
    blob = generate_blob()
    dye_distribution = generate_dye_distribution(blob)
    rle_blob = encode_rle_optimized(blob)
    sparse_dye = encode_sparse_matrix_optimized(dye_distribution)
    with open(os.path.join(data_dir, 'microscope.json'), 'w') as f:
        json.dump(rle_blob.tolist(), f)
    with open(os.path.join(data_dir, 'dye_sensor.json'), 'w') as f:
        json.dump(sparse_dye.tolist(), f)
    decoded_microscope_image = decode_rle_optimized(rle_blob, (height, width))
    decoded_dye_image = decode_sparse_matrix_optimized(sparse_dye, (height, width))
    visualize_image_optimized(decoded_microscope_image, 'Microscope Image', show_image=True, save_image=False)
    visualize_image_optimized(decoded_dye_image, 'Dye Sensor Image', show_image=True, save_image=False)
    cancer_microscope = has_cancer_microscope_optimized(decoded_microscope_image)
    cancer_dye = has_cancer_dye_optimized(decoded_dye_image, decoded_microscope_image) if cancer_microscope else False
    print("Has cancer (Microscope Image):", cancer_microscope)
    print("Has cancer (Dye Sensor Image):", cancer_dye)

if __name__ == "__main__":
    main()
