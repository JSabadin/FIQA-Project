import sys
import os
import cv2
import numpy as np
import torch
import pickle
from PIL import Image
from torchvision.transforms import transforms, v2
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import itertools
from scipy.integrate import quad
from scipy.interpolate import interp1d
from cycler import cycler 
import time

# Add the project root to the system path
project_root = 'C:\\Users\\jerne\\Desktop\\FAKS\\MAG2\\Biometricni_sistemi\\BS_projekt\\experimental_tools'
sys.path.insert(0, project_root)

from face_recognition.load_fr import load_model
from evaluation.edc import *


class PerturbationConfig:
    def __init__(self, type, params=None):
        self.type = type
        self.params = params if params else {}

class Perturbation:
    already_plotted = False
    # already_plotted = True

    @staticmethod
    def plot_images(original_image, perturbed_image, original_title='Original Image', perturbed_title='Perturbed Image'):
        if not Perturbation.already_plotted:
            # Plotting the original image
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(original_image)
            ax.set_title(original_title)
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to remove the border
            plt.show()

            # Plotting the perturbed image
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(perturbed_image)
            ax.set_title(perturbed_title)
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to remove the border
            plt.show()

            Perturbation.already_plotted = True

    @staticmethod
    def gaussian_noise(image, means, vars):
        row, col, ch = image.shape
        gauss = np.random.normal(means, vars ** 0.5, (row, col, ch))
        noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
        Perturbation.plot_images(image, noisy_image, perturbed_title='Gaussian Noise')
        return noisy_image

    @staticmethod
    def salt_and_pepper_noise(image, s_vs_p, amount):
        np.random.seed(41)
        row, col, ch = image.shape
        noisy_image = np.copy(image)

        # Amount of salt and pepper noise
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)

        # Apply salt (white) noise
        coords = [np.random.randint(0, i - 1, num_salt) for i in [row, col]]
        noisy_image[coords[0], coords[1], :] = 255  # Set to white

        # Apply pepper (black) noise
        coords = [np.random.randint(0, i - 1, num_pepper) for i in [row, col]]
        noisy_image[coords[0], coords[1], :] = 0  # Set to black

        Perturbation.plot_images(image, noisy_image, perturbed_title='Salt and Pepper Noise')
        return noisy_image

    @staticmethod
    def multicolored_salt_and_pepper_noise(image, amount):
        row, col, ch = image.shape
        noisy_image = np.copy(image)

        num_noise = int(np.ceil(amount * image.size))
        salt_coords = (np.random.randint(0, row, num_noise), np.random.randint(0, col, num_noise))
        noisy_image[salt_coords] = np.random.randint(0, 255, (num_noise, ch))

        pepper_coords = (np.random.randint(0, row, num_noise), np.random.randint(0, col, num_noise))
        noisy_image[pepper_coords] = np.random.randint(0, 255, (num_noise, ch))

        Perturbation.plot_images(image, noisy_image, perturbed_title='Multicolored Salt and Pepper Noise')
        return noisy_image

    @staticmethod
    def poisson_noise(image):
        # Ensure the image is in float type for scaling
        image = image.astype(np.float32)
        
        # Calculate the scale factor to normalize the image's maximum value to 1
        scale_factor = np.max(image)
        
        # Scale the image to [0, 1] range for Poisson noise since it's intensity dependent
        scaled_image = image / scale_factor
        
        # Generate Poisson noise, the result will be scaled according to the image's intensity
        noisy_image = np.random.poisson(scaled_image * 255) / 255
        
        # Rescale the noisy image back to the original scale
        noisy_image = (noisy_image * scale_factor).clip(0, 255).astype(np.uint8)
        
        # Plot and compare the images
        Perturbation.plot_images(image.astype(np.uint8), noisy_image, perturbed_title='Poisson Noise')
        
        return noisy_image


    @staticmethod
    def add_noise(image, noise_type="gaussian", **params):
        if noise_type == "gaussian":
            return Perturbation.gaussian_noise(image, **params)
        elif noise_type == "salt_and_pepper":
            return Perturbation.salt_and_pepper_noise(image, **params)
        elif noise_type == "multicolored_salt_and_pepper":
            return Perturbation.multicolored_salt_and_pepper_noise(image, **params)
        elif noise_type == "poisson":
            return Perturbation.poisson_noise(image)
        else:
            raise ValueError("Unsupported noise type")
        
    @staticmethod
    def apply_affine_transformation(image, rotation_angle=20, scale=1.2, flip_horizontal=False, flip_vertical=False, shear_x=0, shear_y=0):
        h, w, _ = image.shape
        transformed_image = image.copy()

        # Define the rotation matrix using the specified angle and scale
        M = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, scale)

        # Apply rotation and scaling
        transformed_image = cv2.warpAffine(transformed_image, M, (w, h))

        # Flipping
        if flip_horizontal:
            transformed_image = cv2.flip(transformed_image, 1)  # Flip horizontally
        if flip_vertical:
            transformed_image = cv2.flip(transformed_image, 0)  # Flip vertically

        # Shearing
        if shear_x != 0 or shear_y != 0:
            M_shear = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
            transformed_image = cv2.warpAffine(transformed_image, M_shear, (w, h))

        Perturbation.plot_images(image, transformed_image, perturbed_title='Affine Transformed Image')
        return transformed_image

    
    @staticmethod
    def apply_structured_occlusion(image, border_thickness=30):
        h, w, _ = image.shape

        occluded_image = np.zeros_like(image)
        occluded_image[border_thickness:h - border_thickness, border_thickness:w - border_thickness] = image[border_thickness:h - border_thickness, border_thickness:w - border_thickness]
        
        Perturbation.plot_images(image, occluded_image, perturbed_title='Occluded Image')
        return occluded_image
    
    @staticmethod
    def apply_random_occlusion(image, occlusion_size=30):
        """
        Applies a single random square occlusion to an image.

        :param image: NumPy array representing the image.
        :param occlusion_size: Size of the square occlusion.
        :return: Image with applied occlusion.
        """
        h, w, _ = image.shape
        occluded_image = image.copy()

        # Randomly select the top-left corner of the occlusion
        occlusion_x = random.randint(0, w - occlusion_size)
        occlusion_y = random.randint(0, h - occlusion_size)

        # Apply the occlusion
        occluded_image[occlusion_y:occlusion_y + occlusion_size, occlusion_x:occlusion_x + occlusion_size] = 0  # Setting the block to black
        Perturbation.plot_images(image, occluded_image, perturbed_title='Occluded Image')
        return occluded_image
    
    @staticmethod
    def MixUp(image_array, grid_size=(3, 3)):
        """
        Perform self-mixup on an image by dividing it into a grid and shuffling the pieces,
        ensuring the shuffled image is not identical to the original.

        Parameters:
        image_array (numpy.ndarray): The image to be processed, in NumPy array format.
        grid_size (tuple): The number of segments along the width and height (columns, rows).

        Returns:
        numpy.ndarray: The image after self-mixup, in NumPy array format.
        """
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Image must be a 3-channel (RGB) image")

        height, width, _ = image_array.shape
        grid_w, grid_h = grid_size
        segment_width = width // grid_w
        segment_height = height // grid_h

        # Cut the image into segments and track their original positions
        segments = [(image_array[i*segment_height:(i+1)*segment_height, j*segment_width:(j+1)*segment_width, :], (i, j))
                    for i in range(grid_h) for j in range(grid_w)]

        # Shuffle the segments, ensuring no segment remains in its original position
        while True:
            random.shuffle(segments)
            if all(original_position != (i // grid_w, i % grid_w) for i, (_, original_position) in enumerate(segments)):
                break

        # Create a new image array and paste the shuffled segments
        new_image_array = np.zeros_like(image_array)
        for idx, (segment, _) in enumerate(segments):
            i, j = divmod(idx, grid_w)
            new_image_array[i*segment_height:(i+1)*segment_height, j*segment_width:(j+1)*segment_width, :] = segment
            
        Perturbation.plot_images(image_array, new_image_array, perturbed_title='MixUp Transformed Image')
        return new_image_array
    
    @staticmethod
    def elastic_deformation(image, alpha, sigma):
        """
        Apply ElasticTransform from torchvision to a NumPy image.

        Parameters:
        image (numpy.ndarray): The input image array of shape (H, W, C) or (H, W) for grayscale.
        alpha (float): The scaling factor for deformation intensity.
        sigma (float): The smoothing factor for the displacement fields.

        Returns:
        numpy.ndarray: Transformed image array of shape (H, W, C) or (H, W) for grayscale.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")

        # Convert the NumPy array to PIL Image
        if image.ndim == 2:  # Grayscale
            pil_image = Image.fromarray(image)
        elif image.ndim == 3 and image.shape[2] in [3, 4]:  # Color Image
            pil_image = Image.fromarray(image, 'RGB' if image.shape[2] == 3 else 'RGBA')
        else:
            raise ValueError("Unsupported image shape")

        # Apply ElasticTransform
        transform = transforms.ElasticTransform(alpha=alpha, sigma=sigma, interpolation=F.InterpolationMode.BILINEAR, fill=1)
        transformed_pil_image = transform(pil_image)

        # Convert back to NumPy array
        transformed_numpy_image = np.array(transformed_pil_image)
        Perturbation.plot_images(image, transformed_numpy_image, perturbed_title='Elastic Transformed Image')
        return transformed_numpy_image


class FaceEmbeddingModel:
    def __init__(self, model_name):
        self.model, self.trans = load_model(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def extract_embeddings(self, images):
        batch = torch.stack([self.trans(Image.fromarray(img)) for img in images])
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = embeddings.cpu() if embeddings.is_cuda else embeddings
        return embeddings.numpy()

class QualityAssessment:
    @staticmethod
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

def process_image(image_paths, model, embeddings, quality_scores, perturbation_type, perturbation_params=None):
    original_imgs = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
    perturbed_imgs = []
    processing_times = []

    for original_img in original_imgs:
        start_time = time.time()
        if perturbation_type == "structured_occlusion":
            perturbed_img = Perturbation.apply_structured_occlusion(original_img, **perturbation_params)
        elif perturbation_type ==  'random_occlusion':
            perturbed_img = Perturbation.apply_random_occlusion(original_img, **perturbation_params)
        elif perturbation_type == 'mixup':
            perturbed_img = Perturbation.MixUp(original_img, **perturbation_params)
        elif perturbation_type == 'elastic':
            perturbed_img = Perturbation.elastic_deformation(original_img, **perturbation_params)
        elif perturbation_type == "affine":
            perturbed_img = Perturbation.apply_affine_transformation(original_img, **perturbation_params)
        else:
            perturbed_img = Perturbation.add_noise(original_img, noise_type=perturbation_type, **perturbation_params)
        perturbed_imgs.append(perturbed_img)

    original_embeddings = model.extract_embeddings(original_imgs)
    perturbed_embeddings = model.extract_embeddings(perturbed_imgs)

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        quality_score = QualityAssessment.cosine_similarity(original_embeddings[i], perturbed_embeddings[i])
        processing_time = time.time() - start_time

        processing_times.append(processing_time)
        embeddings[image_name] = original_embeddings[i].tolist()
        quality_scores[image_name] = quality_score

        print(f"Processed {image_name}; Quality Score = {quality_score}")
    return processing_times
def process_dataset(dataset_path, model, perturbation_type, perturbation_params, unique_filenames, batch_size=512):
    embeddings = {}
    quality_scores = {}
    all_processing_times = []

    all_image_paths = [os.path.join(root, file) for root, _, files in os.walk(dataset_path) for file in files if file.endswith('.jpg') and os.path.basename(file) in unique_filenames]

    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i:i + batch_size]
        batch_times = process_image(batch_paths, model, embeddings, quality_scores, perturbation_type, perturbation_params)
        all_processing_times.extend(batch_times)

    # Calculating mean and variance of processing times
    mean_processing_time = np.mean(all_processing_times)
    variance_processing_time = np.var(all_processing_times)

    print(f"Mean Processing Time: {mean_processing_time} seconds")
    print(f"Variance of Processing Time: {variance_processing_time} seconds^2")

    return embeddings, quality_scores

def save_results(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

# Function to load a pickle file
def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
# Function to find all .pkl files containing 'xqlfw' in their filename within a specified directory
def find_pkl_files(directory, keyword):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl') and keyword in file:
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def parse_verification_protocol_xqlfw(file_path):
    verification_pairs = []

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    for line in lines:
        elements = line.split()
        
        if len(elements) == 3:
            person, img1, img2 = elements
            image1 = f"{person}_{int(img1):04d}.jpg"
            image2 = f"{person}_{int(img2):04d}.jpg"
            verification_pairs.append((image1, image2, 1))
        
        elif len(elements) == 4:
            person1, img1, person2, img2 = elements
            image1 = f"{person1}_{int(img1):04d}.jpg"
            image2 = f"{person2}_{int(img2):04d}.jpg"
            verification_pairs.append((image1, image2, 0))
        
        else:
            print(f"Warning: Skipping malformed line: {line}")

    return verification_pairs

def parse_verification_protocol_cplfw(file_path):
    verification_pairs = []

    with open(file_path, 'r') as file:
        while True:
            line1 = file.readline().strip()
            line2 = file.readline().strip()

            # Break the loop if either line is empty (end of file)
            if not line1 or not line2:
                break

            # Extract image names and labels
            image1, label1 = line1.split()
            image2, label2 = line2.split()

            # Assuming label1 and label2 are the same for a valid pair
            if label1 == label2:
                # Pair the images with the label (1 for positive, 0 for negative)
                verification_pairs.append((image1, image2, int(label1)))
            else:
                print(f"Warning: Mismatched labels in consecutive lines: {line1} and {line2}")

    return verification_pairs

def parse_verification_protocol_adience(dataset_path):
    # Set a seed for reproducibility
    random.seed(14)  # You can use any number as the seed
    
    # Get the list of all directories
    dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    positive_pairs = []
    negative_pairs = []

    while len(positive_pairs) < 4000 or len(negative_pairs) < 4000:
        for d in dirs:
            # Get all images for this individual
            images = [img for img in os.listdir(os.path.join(dataset_path, d))]
            
            # Group images by individual
            individuals = {}
            for image in images:
                individual = image.split('.')[1]  # Adjust this based on your naming convention
                if individual not in individuals:
                    individuals[individual] = []
                individuals[individual].append(image)
            
            # Generate all positive pairs for each individual
            for individual_images in individuals.values():
                pairs = list(itertools.combinations(individual_images, 2))
                # Add these pairs to the positive_pairs list until it has 3000 pairs
                for pair in pairs:
                    if len(positive_pairs) < 4000 and pair not in positive_pairs:
                        positive_pairs.append((*pair, 1))
            
            # Pair this image with one image from every other individual to create negative pairs
            for other_d in dirs:
                if other_d != d and len(negative_pairs) < 4000:
                    other_images = [img for img in os.listdir(os.path.join(dataset_path, other_d))]
                    other_image = random.choice(other_images)
                    # Add this pair to the negative_pairs list until it has 3000 pairs
                    if (image, other_image, 0) not in negative_pairs:
                        negative_pairs.append((image, other_image, 0))

    # Combine positive and negative pairs
    dataset = positive_pairs + negative_pairs

    return dataset
