import os
from dotenv import load_dotenv
from huggingface_hub import login

def hf_login(env_key: str = "HF_TOKEN") -> None:
    """
    Logs in to Hugging Face using a token stored in a .env file.

    Args:
        env_key (str): Environment variable name for the HF token.
                       Default is 'HF_TOKEN'.
    """
    # Load .env only once (safe to call multiple times)
    load_dotenv("my.env")

    token = os.getenv(env_key)

    if not token:
        raise ValueError(f"{env_key} not found in .env file")

    login(token=token)
