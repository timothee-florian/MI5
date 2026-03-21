from transformers import AutoModelForCausalLM, AutoTokenizer

# Point to your local directory
local_dir = "./qwen_model"

# Load from local directory
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    torch_dtype="auto",
    device_map="auto"
)

# Use it normally
messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    # {"role": "user", "content": "Hello!"}
    {"role": "user", "content": "What is the capital of France?"}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)[0]

print(response)