import os
from openai import OpenAI
from config import REASONING_MODEL, NEBIUS_API_KEY

# Fixed the missing 'os' and defined the client properly
client_nebius = OpenAI(
    base_url ="https://api.studio.nebius.ai/v1",
    api_key =NEBIUS_API_KEY
)

def get_causal_knowledge(user_desc, captions):
    prompt = f"Context: {user_desc}\nVisuals: {captions}\nIdentify the cause and effect (CR0)."
    
    response = client_nebius.chat.completions.create(
        model=REASONING_MODEL, 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content