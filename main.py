import os
import json
from src.encoding import encode_rle, encode_sparse_matrix
from src.decoding import decode_rle, decode_sparse_matrix
from src.generation import generate_blob, generate_dye_distribution
from src.visualization import visualize_image
from src.cancer_detection import has_cancer_dye, has_cancer_microscope

# Define image dimensions
WIDTH, HEIGHT = 1000, 1000

# Create the data directory if it doesn't exist
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Generate simulated images
blob = generate_blob()
dye_distribution = generate_dye_distribution(blob)

# Encode the simulated images using the chosen data structures
rle_blob = encode_rle(blob)
sparse_dye = encode_sparse_matrix(dye_distribution)

# Save the encoded images
with open(os.path.join(data_dir, 'microscope.json'), 'w') as f:
    json.dump(rle_blob.tolist(), f)

with open(os.path.join(data_dir, 'dye_sensor.json'), 'w') as f:
    json.dump(sparse_dye.tolist(), f)

# Decode the images
decoded_microscope_image = decode_rle(rle_blob, (HEIGHT, WIDTH))
decoded_dye_image = decode_sparse_matrix(sparse_dye, (HEIGHT, WIDTH))

# Visualize the decoded images
visualize_image(decoded_microscope_image, 'Microscope Image')
visualize_image(decoded_dye_image, 'Dye Sensor Image')

cancer_microscope = has_cancer_microscope(decoded_microscope_image)

# If microscope image doesn't have cancer, no need to check dye sensor image
if not cancer_microscope:
    cancer_dye = False
else:
    cancer_dye = has_cancer_dye(decoded_dye_image, decoded_microscope_image)

print("Has cancer (Microscope Image):", cancer_microscope)
print("Has cancer (Dye Sensor Image):", cancer_dye)
