from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load pre-trained models and tokenizers
def load_models():
    # Image captioning model
    image_caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
    image_caption_model = VisionEncoderDecoderModel.from_pretrained(image_caption_model_name)
    image_caption_tokenizer = AutoTokenizer.from_pretrained(image_caption_model_name)

    # Text embedding model
    text_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_model = AutoModel.from_pretrained(text_embedding_model_name)
    text_embedding_tokenizer = AutoTokenizer.from_pretrained(text_embedding_model_name)

    return image_caption_model, image_caption_tokenizer, text_embedding_model, text_embedding_tokenizer


# Load models
image_caption_model, image_caption_tokenizer, text_embedding_model, text_embedding_tokenizer = load_models()


# Function to predict image captions
def predict_image_caption(image_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_caption_model.to(device)

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long, device=device)
    output_ids = image_caption_model.generate(pixel_values, attention_mask=attention_mask, max_length=16, num_beams=4)
    preds = image_caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return preds


# Function to embed text
def embed_text(texts):
    inputs = text_embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = text_embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


# Index images with their captions
def index_images(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                   img.endswith(('.png', '.jpg', '.jpeg'))]
    captions = predict_image_caption(image_paths)
    caption_embeddings = embed_text(captions)
    return image_paths, captions, caption_embeddings


# Search for images using keywords
def search_images(keyword, image_paths, captions, caption_embeddings):
    keyword_embedding = embed_text([keyword])
    similarities = cosine_similarity(keyword_embedding, caption_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]  # Sort indices by similarity in descending order
    return [(image_paths[i], captions[i], similarities[i]) for i in sorted_indices]


# Example usage
image_folder = 'Test_Data'
image_paths, captions, caption_embeddings = index_images(image_folder)


# keyword = "yellow jacket"
keyword = input("search: ")
results = search_images(keyword, image_paths, captions, caption_embeddings)

# print("Search results:")
# for i, (img_path, caption, similarity) in enumerate(results[:5]):  # Display top 5 results
#     print(f"Image: {img_path}, Caption: {caption}, Similarity: {similarity:.4f}")
#     img = Image.open(img_path)
#     img.show()

print("Search results:")
for i, (img_path, caption, similarity) in enumerate(results[:5]):  # Display top 5 results
    if similarity > 0.35:
        img = Image.open(img_path)
        img.show()

