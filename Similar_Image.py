import os
import cv2
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from sklearn.metrics.pairwise import cosine_similarity


def load_images_from_directory(directory):
    image_files = os.listdir(directory)
    images = []
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = cv2.imread(image_path)
        images.append((image_file, image))
    return images


def extract_resnet50_features(images):
    base_model = ResNet50(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    feature_vectors = [model.predict(preprocess_input(cv2.resize(image, (224, 224))).reshape(1, 224, 224, 3)).flatten()
                       for _, image in images]
    return feature_vectors


def detect_similar_images(image_files, feature_vectors):
    similar_images = []
    processed_images = set()  # Store processed image filenames
    num_images = len(image_files)
    for i in range(num_images):
        if image_files[i] not in processed_images:  # Check if the image is already processed
            group = []
            for j in range(num_images):
                if i != j:
                    similarity_score = cosine_similarity([feature_vectors[i]], [feature_vectors[j]])[0][0]
                    if similarity_score > 0.80:
                        group.append(image_files[j])
                        processed_images.add(image_files[j])
            if group:
                group.append(image_files[i])
                similar_images.append(group)
                processed_images.add(image_files[i])
    return similar_images


image_dir = "Test_Data"
images = load_images_from_directory(image_dir)
feature_vectors = extract_resnet50_features(images)
similar_image_groups = detect_similar_images([image[0] for image in images], feature_vectors)

for group in similar_image_groups:
    print("Similar images: ", ", ".join(group))
