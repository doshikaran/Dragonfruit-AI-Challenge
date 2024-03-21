import os
import matplotlib.pyplot as plt

def visualize_image_optimized(image, title, output_dir=None, filename=None, show_image=True, save_image=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    if save_image and output_dir is not None:
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        if filename is None:
            filename = 'image.png'
        file_path = os.path.join(images_dir, filename)
        plt.savefig(file_path)
        print(f"Image saved at {file_path}")
    if show_image:
        plt.show()
