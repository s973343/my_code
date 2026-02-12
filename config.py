import os
from dotenv import load_dotenv

# Load my.env from the same directory as this config file
base_dir = os.path.dirname(__file__)
env_path = os.path.join(base_dir, "my.env")
load_dotenv(env_path)

# Paths
VIDEO_INPUT = "data/Trishul_480P.mp4"
AUDIO_OUTPUT = "data/temp_audio.mp3"
FRAME_DIR = "data/frames/"
DB_PATH = "./video_db"
#Huggingface dataset
DATASET_NAME = "TIGER-Lab/VISTA-400K"

# Model Configs
CLIP_MODEL = "ViT-B/16" # 512-dim
WHISPER_MODEL = "medium"
REASONING_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct" # DeepSeek V3.2 logic
VLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct" # Groq
#---------------------#NEBIUS models----------------------
#VL_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
USER_DESCRIPTION = "A video showing a movie schene"

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
if not NEBIUS_API_KEY:
    raise ValueError("[ERROR] NEBIUS_API_KEY is missing! Please create a .env file in this folder with: NEBIUS_API_KEY=your_key_here")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[WARN] GROQ_API_KEY is missing. RAG Query features might fail.")
