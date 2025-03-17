import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path

# -------------------------------
# Image Preprocessing Functions
# -------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # Generate possible aspect-ratio slices
    target_ratios = set((i, j) for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num)
    target_aspect_ratio = min(target_ratios, key=lambda x: abs((x[0]/x[1]) - aspect_ratio))

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    # Slice the resized image into multiple squares of size (image_size x image_size)
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        ))
        for i in range(target_aspect_ratio[0] * target_aspect_ratio[1])
    ]

    if use_thumbnail and len(processed_images) != 1:
        # Optionally add a thumbnail version
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values

# ------------------------------------------------
# Prompt for Store Cleanliness ("Yes"/"No" output)
# ------------------------------------------------
def generate_store_cleanliness_prompt():
    """
    Creates a prompt that asks the model to analyze a clothing store image
    and provide 'Yes' or 'No' answers to specific cleanliness/organization criteria.
    """
    return """
Please analyze the attached image of a clothing store and answer “Yes” or “No” to the following questions:

1. Are the items neatly folded and arranged?
2. Is the floor clear and unobstructed?
4. Is the signage and pricing displayed in a way that does not create clutter?
5. Are the surfaces (shelves, floor, etc.) visibly clean?

Based on these answers, does the store appear clean and organized overall? Answer “Yes” or “No”.
"""

# -------------------------------
# Model Setup
# -------------------------------
model_name = "OpenGVLab/InternVL2_5-8B"
print("Loading the model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # 1. Load the image
    image_file = "/home/azureuser/workspace/Genfied/images/cam1/frame_0000.jpg"  # Replace with your store image path
    pixel_values = load_image(image_file).to(torch.float16).to(device)

    # 2. Generate the prompt
    store_prompt = generate_store_cleanliness_prompt()

    # 3. Prepare the full question for the model
    #    "<image>" token indicates to the model that the attached pixel_values is relevant
    question = f"<image>\n{store_prompt}"
    generation_config = dict(max_new_tokens=512, do_sample=True)

    # 4. Run the model
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    # 5. Print the response
    print(f"Prompt:\n{question}\n")
    print(f"Model's Response:\n{response}")
