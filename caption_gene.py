from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image


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
def predict_image_caption(image_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    attention_mask = torch.ones(pixel_values.shape[:2], dtype=torch.long, device=device)
    output_ids = image_caption_model.generate(pixel_values, attention_mask=attention_mask, max_length=16, num_beams=4)
    preds = image_caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return preds


# Example usage
image_paths = ['Test_Data/IMG_20201013_131057.jpg']
predicted_captions = predict_image_caption(image_paths)
input_caption = predicted_captions[-1]  # Selecting the first caption from the list
print("Original Caption:", input_caption)
