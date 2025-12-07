from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import main_dir
import os
model_name = "BAAI/bge-base-en-v1.5"#"Qwen/Qwen2.5-0.5B-Instruct"
local_dir = os.path.join(main_dir, model_name)  # Local directory to save the model

if __name__ == '__main__':
    # Download and save
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print(f"Model saved to {local_dir}")