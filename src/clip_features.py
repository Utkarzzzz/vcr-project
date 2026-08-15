import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    @torch.no_grad()
    def encode_pil(self, img):
        img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        feat = self.model.get_image_features(**inputs).pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy()

    def encode_image(self, image_path):
        return self.encode_pil(Image.open(image_path))

    @torch.no_grad()
    def encode_texts(self, texts):
        # CLIP's text encoder caps at 77 tokens; long rationales get cut,
        # which is a known limitation of this simple approach.
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        feat = self.model.get_text_features(**inputs).pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()


def pair_features(img_vec, txt_vec):
    # Given a normalized image vector and text vector, build one feature
    # row the classifier can learn from. The element-wise product and the
    # cosine similarity are what let a linear model judge "do these match".
    cos = float(np.dot(img_vec, txt_vec))
    return np.concatenate([img_vec, txt_vec, img_vec * txt_vec, [cos]])
