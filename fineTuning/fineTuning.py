import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator

# Model path and configurations
model_name = r"C:\Users\Google Tech\Desktop\Meta-Llama-3.1-8B-Instruct"
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True
device_map = {"": 0}
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
)

# Initialize accelerator
accelerator = Accelerator()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model and tokenizer for GPU
model = accelerator.prepare(model)

# Define the input sentence
input_sentence = "write a small letter to a friend, today is his birthday"

# Tokenize and move input to GPU
inputs = tokenizer(input_sentence, return_tensors="pt").to(accelerator.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)

# Decode the output
output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {input_sentence}")
print(f"Output: {output_sentence}")
