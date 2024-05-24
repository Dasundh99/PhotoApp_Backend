from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
import os
import random
import concurrent.futures

app = Flask(__name__)

# Define constants
No_of_test_samples = 6
No_of_best_looking_samples = 60
ssim_threshold = 12.5
symmetry_threshold = 0.55
color_variation_threshold = 80
blur_threshold = 2000000

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to resize images to a specified size
def resize_image(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# Load reference images and resize them
reference_folder = 'Best_Looking_Samples'
reference_images = os.listdir(reference_folder)
reference_images = random.sample(reference_images, k=No_of_best_looking_samples)

resized_reference_images = {}
reference_grayscale_images = {}
for ref_image in reference_images:
    reference_image = cv2.imread(os.path.join(reference_folder, ref_image), cv2.IMREAD_GRAYSCALE)
    ref_height, ref_width = reference_image.shape[:2]
    resized_reference_images[ref_image] = resize_image(reference_image, (ref_width // 2, ref_height // 2))
    reference_grayscale_images[ref_image] = reference_image


# Function to process each device photo
def process_photo(photo_path):
    device_image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    photo = os.path.basename(photo_path)

    # Calculate blur score
    gradient_magnitude = cv2.Laplacian(device_image, cv2.CV_64F).var()

    # Reject further analysis if the image is blurry
    if gradient_magnitude >= blur_threshold:
        return None

    # Initialize dictionary to store SSIM scores for each device photo
    device_scores = defaultdict(float)
    # Initialize list to store symmetry scores for each device photo
    symmetry_scores = []
    # Initialize dictionary to store color variation scores for each device photo
    color_variation_scores = {}

    # Calculate SSIM score
    for ref_image, reference_image in reference_grayscale_images.items():
        resized_reference_image = resized_reference_images[ref_image]
        score = ssim(resized_reference_image, resize_image(device_image, resized_reference_image.shape[::-1]))
        device_scores[photo] += score

    # Calculate symmetry score
    flipped = cv2.flip(device_image, 1)
    symmetry = ssim(device_image, flipped)
    symmetry_scores.append((photo, symmetry))

    # Calculate color variation score
    lab_image = cv2.cvtColor(cv2.imread(photo_path), cv2.COLOR_BGR2LAB)
    std_dev = np.std(lab_image, axis=(0, 1))
    color_variation_scores[photo] = np.sum(std_dev)

    return {
        "photo": photo,
        "ssim_score": device_scores[photo],
        "symmetry_score": symmetry,
        "color_variation_score": color_variation_scores[photo]
    }


@app.route('/upload', methods=['POST'])
def upload_images():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    image_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        image_paths.append(file_path)

    # Process images concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_photo, path) for path in image_paths]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    if not results:
        return jsonify({"error": "No images passed the blur threshold"}), 400

    # Rank the device photos based on the sum of SSIM scores
    ranked_device_photos = sorted(results, key=lambda x: x['ssim_score'], reverse=True)
    # Rank the symmetric photos based on the symmetry scores
    ranked_symmetry_device_photos = sorted(results, key=lambda x: x['symmetry_score'], reverse=True)

    # List to store images surpassing all thresholds
    images_surpassing_thresholds = []

    for photo_result in ranked_device_photos:
        photo = photo_result['photo']
        score_sum = photo_result['ssim_score']
        symmetry = photo_result['symmetry_score']
        variation_score = photo_result['color_variation_score']

        if (score_sum > ssim_threshold and
                symmetry > symmetry_threshold and
                variation_score > color_variation_threshold):
            images_surpassing_thresholds.append(photo)

    return jsonify({"images_surpassing_thresholds": images_surpassing_thresholds})


if __name__ == '__main__':
    app.run(debug=True)
