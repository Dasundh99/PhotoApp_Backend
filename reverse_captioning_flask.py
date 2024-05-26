from flask import Flask, request, jsonify, send_from_directory
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoModel, AutoTokenizer
import torch
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load pre-trained models and tokenizers
def load_models():
    image_caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
    image_caption_model = VisionEncoderDecoderModel.from_pretrained(image_caption_model_name)
    image_caption_tokenizer = AutoTokenizer.from_pretrained(image_caption_model_name)

    text_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_model = AutoModel.from_pretrained(text_embedding_model_name)
    text_embedding_tokenizer = AutoTokenizer.from_pretrained(text_embedding_model_name)

    return image_caption_model, image_caption_tokenizer, text_embedding_model, text_embedding_tokenizer


image_caption_model, image_caption_tokenizer, text_embedding_model, text_embedding_tokenizer = load_models()

indexed_data = {'image_paths': [], 'captions': [], 'caption_embeddings': []}
image_folder = ""


@app.route('/set_image_folder', methods=['POST'])
def set_image_folder():
    global image_folder
    data = request.get_json()
    image_folder = data['image_folder']
    return jsonify({"message": "Image folder set successfully"}), 200


@app.route('/index_images', methods=['POST'])
def index_images():
    global image_folder
    if not image_folder:
        return jsonify({"error": "Image folder is not set"}), 400

    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                   img.endswith(('.png', '.jpg', '.jpeg'))]
    captions = predict_image_caption(image_paths)
    caption_embeddings = embed_text(captions)

    indexed_data['image_paths'] = image_paths
    indexed_data['captions'] = captions
    indexed_data['caption_embeddings'] = caption_embeddings.numpy().tolist()

    return jsonify({"message": "Images indexed successfully"}), 200


@app.route('/search_images', methods=['GET'])
def search_images():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({"error": "Keyword is required"}), 400

    if not indexed_data['caption_embeddings']:
        return jsonify({"error": "No indexed data available. Please index images first."}), 400

    keyword_embedding = embed_text([keyword])
    similarities = cosine_similarity(keyword_embedding, np.array(indexed_data['caption_embeddings']))[0]
    sorted_indices = np.argsort(similarities)[::-1]

    results = [(os.path.basename(indexed_data['image_paths'][i]), indexed_data['captions'][i], similarities[i]) for i in sorted_indices]

    return jsonify(results[:5]), 200


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(image_folder, filename)



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


def embed_text(texts):
    inputs = text_embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = text_embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
