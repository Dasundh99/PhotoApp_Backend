from flask import Flask, request, jsonify
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)


# Load pre-trained models and tokenizers
def load_models():
    # Image captioning model
    image_caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
    image_caption_model = VisionEncoderDecoderModel.from_pretrained(image_caption_model_name)
    image_caption_tokenizer = AutoTokenizer.from_pretrained(image_caption_model_name)

    return image_caption_model, image_caption_tokenizer


# Load models
image_caption_model, image_caption_tokenizer = load_models()


# Function to predict image captions
def predict_image_caption(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    image = image.convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
    attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long, device=device)
    output_ids = image_caption_model.generate(pixel_values, attention_mask=attention_mask, max_length=16, num_beams=4)
    preds = image_caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return preds[0]


# Define the route for the API
@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    caption = predict_image_caption(image)

    return jsonify({'caption': caption})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
