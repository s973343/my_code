import os
import base64
from openai import OpenAI
from config import REASONING_MODEL, NEBIUS_API_KEY

# Initialize Nebius Client
client_nebius = OpenAI(
    base_url="https://api.studio.nebius.ai/v1",
    api_key=NEBIUS_API_KEY
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_captions(frame_path):
    base64_image = encode_image(frame_path)
    
    # We use Qwen2-VL or similar vision model on Nebius
    response = client_nebius.chat.completions.create(
        model=REASONING_MODEL, # Using the Qwen VL model defined in config
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Describe this video frame. Provide two levels of detail:\n"
                                "1. SHORT: A 10-20 word summary.\n"
                                "2. LONG: A 60-70 word descriptive paragraph.\n"
                                "Return only the labels SHORT: and LONG: followed by the text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ],
        max_tokens=300
    )
    
    raw_text = response.choices[0].message.content
    
    # Parsing logic to separate Short and Long captions
    try:
        short_cap = raw_text.split("SHORT:")[1].split("LONG:")[0].strip()
        long_cap = raw_text.split("LONG:")[1].strip()
    except IndexError:
        # Fallback if the model doesn't follow formatting perfectly
        short_cap = raw_text[:50]
        long_cap = raw_text
        
    return short_cap, long_cap