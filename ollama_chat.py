import ollama
from pathlib import Path

def analyze_store_image(input_image_path):
    if isinstance(input_image_path, Path):
        input_image_path = str(input_image_path)
    
    store_prompt = """
    Please analyze the attached image of a clothing store and answer "Yes" or "No" to the following questions:

    1. Are the items neatly folded and arranged?
    2. Is the floor clear and unobstructed
    3. Is the signage and pricing displayed in a way that does not create clutter?
    4. Are the surfaces (shelves, floor, cash counter, etc.) visibly clean?
    5. Is the billing counter clean?

    Based on these answers, does the store appear clean and organized overall? Answer "Yes" or "No"
    """

    response = ollama.chat(
        model = "qwen2.5:7b",
        messages = [
            {
                "role":"user",
                "content": store_prompt,
                "images": [input_image_path] if input_image_path else []
            }
        ]
    )

    return response.get("message", {}).get("content","").strip()

if __name__ == '__main__':
    image_path = "/home/azureuser/workspace/Genfied/images/cam1/frame_0000.jpg"
    result = analyze_store_image(image_path)
    print(result)