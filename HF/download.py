from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
local_dir = "./qwen_model"  # Local directory to save the model

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)

print(f"Model saved to {local_dir}")