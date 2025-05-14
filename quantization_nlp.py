##------------------------------------------------------------------------------------------------------------------------------##
##------------------------------------------------Quantisation from sratch------------------------------------------------------##
##------------------------------------------------------------------------------------------------------------------------------##
import os
import time
import torch
import copy
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Define constants
MODEL_NAME = "gpt2"
DATASET_NAME = "ptb_text_only"
QUANTIZED_MODELS_DIR = "./quantized_models"

# Create directory for saving models
os.makedirs(QUANTIZED_MODELS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="test")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load original model
print("Loading original model...")
original_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
original_model.resize_token_embeddings(len(tokenizer))

# Helper: Calculate model memory usage
def get_memory_usage(model):
    return sum([param.numel() * param.element_size() for param in model.parameters()]) / (1024 ** 2)

# Helper: Measure inference latency
def measure_latency(model, tokenized_input):
    model.eval()
    device = next(model.parameters()).device
    inputs = torch.tensor(tokenized_input).unsqueeze(0).to(device)
    start_time = time.time()
    with torch.no_grad():
        model.generate(inputs, max_new_tokens=20, pad_token_id=tokenizer.pad_token_id)
    return time.time() - start_time

# Helper: Calculate perplexity
def calculate_perplexity(model, tokenized_dataset):
    print("Calculating perplexity...")
    model.eval()
    total_loss = 0
    total_words = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(tokenized_dataset):
            inputs = torch.tensor(batch["input_ids"]).to(device).unsqueeze(0)
            outputs = model(input_ids=inputs, labels=inputs)
            total_loss += outputs.loss.item() * inputs.size(1)
            total_words += inputs.size(1)
    if total_words == 0:
        raise ValueError("No valid sentences for perplexity calculation.")
    return math.exp(total_loss / total_words)

# Quantization: Helper for scale and zero point
def calculate_scale_and_zero_point(min_val, max_val, dtype=torch.int8):
    qmin, qmax = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    return scale, zero_point

def quantize_tensor(tensor, scale, zero_point):
    return ((tensor / scale) + zero_point).round().to(torch.int8)

def dequantize_tensor(quantized_tensor, scale, zero_point):
    return (quantized_tensor.float() - zero_point) * scale

# Whole-model quantization
def quantize_whole_model(model):
    print("Quantizing the entire model...")
    quantized_model = copy.deepcopy(model)
    quantized_params = {}
    for name, param in quantized_model.named_parameters():
        min_val, max_val = param.data.min(), param.data.max()
        scale, zero_point = calculate_scale_and_zero_point(min_val, max_val)
        quantized_param = quantize_tensor(param.data, scale, zero_point)
        quantized_params[name] = (quantized_param, scale, zero_point)
    print("Whole-model quantization complete.")
    return quantized_model, quantized_params

# Selective component quantization
def quantize_selected_components(model, layers_to_quantize):
    print(f"Quantizing selective layers: {layers_to_quantize}")
    quantized_model = copy.deepcopy(model)
    quantized_params = {}
    for name, param in quantized_model.named_parameters():
        if any(layer in name for layer in layers_to_quantize):
            print(f"Quantizing layer: {name}")
            min_val, max_val = param.data.min(), param.data.max()
            scale, zero_point = calculate_scale_and_zero_point(min_val, max_val)
            quantized_param = quantize_tensor(param.data, scale, zero_point)
            quantized_params[name] = (quantized_param, scale, zero_point)
    print("Selective quantization complete.")
    return quantized_model, quantized_params

# Evaluate original model
print("Evaluating original model...")
original_memory = get_memory_usage(original_model)
original_latency = measure_latency(original_model, tokenized_dataset[0]["input_ids"])
original_perplexity = calculate_perplexity(original_model, tokenized_dataset)
print(f"Original Model -> Memory: {original_memory:.2f}MB, Latency: {original_latency:.4f}s, Perplexity: {original_perplexity:.4f}")

# Apply whole-model quantization
quantized_whole_model, quantized_whole_params = quantize_whole_model(original_model)

# Evaluate whole-model quantization
print("Evaluating whole-quantized model...")
quantized_whole_memory = get_memory_usage(quantized_whole_model)
quantized_whole_latency = measure_latency(quantized_whole_model, tokenized_dataset[0]["input_ids"])
quantized_whole_perplexity = calculate_perplexity(quantized_whole_model, tokenized_dataset)
print(f"Whole-Quantized Model -> Memory: {quantized_whole_memory:.2f}MB, Latency: {quantized_whole_latency:.4f}s, Perplexity: {quantized_whole_perplexity:.4f}")

# Apply selective quantization
layers_to_quantize = ["attn", "ffn"]
quantized_selective_model, quantized_selective_params = quantize_selected_components(original_model, layers_to_quantize)

# Evaluate selective quantization
print("Evaluating selectively quantized model...")
quantized_selective_memory = get_memory_usage(quantized_selective_model)
quantized_selective_latency = measure_latency(quantized_selective_model, tokenized_dataset[0]["input_ids"])
quantized_selective_perplexity = calculate_perplexity(quantized_selective_model, tokenized_dataset)
print(f"Selective-Quantized Model -> Memory: {quantized_selective_memory:.2f}MB, Latency: {quantized_selective_latency:.4f}s, Perplexity: {quantized_selective_perplexity:.4f}")

# Save models
print("Saving models...")
torch.save(original_model.state_dict(), f"{QUANTIZED_MODELS_DIR}/original_model_weights.pth")
torch.save(quantized_whole_model.state_dict(), f"{QUANTIZED_MODELS_DIR}/quantized_whole_model_weights.pth")
torch.save(quantized_selective_model.state_dict(), f"{QUANTIZED_MODELS_DIR}/quantized_selective_model_weights.pth")

# Visualization
metrics = ["Memory (MB)", "Latency (s)", "Perplexity"]
original_values = [original_memory, original_latency, original_perplexity]
quantized_whole_values = [quantized_whole_memory, quantized_whole_latency, quantized_whole_perplexity]
quantized_selective_values = [quantized_selective_memory, quantized_selective_latency, quantized_selective_perplexity]

plt.figure(figsize=(12, 6))
x = range(len(metrics))
plt.bar(x, original_values, width=0.2, label="Original Model", align="center")
plt.bar([p + 0.2 for p in x], quantized_whole_values, width=0.2, label="Whole-Quantized", align="center")
plt.bar([p + 0.4 for p in x], quantized_selective_values, width=0.2, label="Selective-Quantized", align="center")
plt.xticks([p + 0.2 for p in x], metrics)
plt.ylabel("Values")
plt.title("Comparison of Original and Quantized Models")
plt.legend()
plt.show()


##------------------------------------------------------------------------------------------------------------------------------##
##--------------------------------------Bitsandbytes Integration and NF4 Quantization-------------------------------------------##
##------------------------------------------------------------------------------------------------------------------------------##

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import bitsandbytes as bnb
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm

# Define constants
MODEL_NAME = "gpt2"  # Replace with your desired model
DATASET_NAME = "wikitext"  # Using WikiText dataset as per assignment
DATASET_CONFIG = "wikitext-2-raw-v1"  # WikiText raw version for perplexity
QUANTIZED_MODELS_DIR = "./quantized_models"  # Directory to save quantized models

# Create directory for saving models
os.makedirs(QUANTIZED_MODELS_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token

# Load WikiText dataset
print("Loading WikiText dataset...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")

# Prepare data for evaluation (filter non-empty sentences and limit to 3000 examples)
print("Preparing data for evaluation...")
data = [entry.strip() for entry in dataset["text"] if entry.strip()]
data = data[:3000]

# Helper functions
def get_model_memory(model):
    """Calculate memory usage of a model in MB."""
    return sum([param.element_size() * param.numel() for param in model.parameters()]) / (1024 ** 2)

def measure_latency(model, tokenizer, text, device):
    """Measure inference latency for the model."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start_time = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)
    end_time = time.time()
    return end_time - start_time

def calculate_perplexity(model, tokenizer, data, device):
    """Calculate perplexity for a dataset."""
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for text in tqdm(data, desc="Calculating perplexity"):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=tokenizer.model_max_length
            ).to(device)
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            total_loss += loss * labels.size(1)
            total_words += labels.size(1)
    return math.exp(total_loss / total_words)

# Load and evaluate original FP32 model
print("Loading and evaluating original FP32 model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
memory_fp32 = get_model_memory(model_fp32)
latency_fp32 = measure_latency(model_fp32, tokenizer, "The future of AI is", device)
perplexity_fp32 = calculate_perplexity(model_fp32, tokenizer, data, device)
print(f"FP32 -> Memory: {memory_fp32:.2f} MB, Latency: {latency_fp32:.4f}s, Perplexity: {perplexity_fp32:.4f}")

# Load and evaluate 8-bit quantized model
print("Applying 8-bit quantization using Bitsandbytes...")
model_8bit = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
memory_8bit = get_model_memory(model_8bit)
latency_8bit = measure_latency(model_8bit, tokenizer, "The future of AI is", device)
perplexity_8bit = calculate_perplexity(model_8bit, tokenizer, data, device)
print(f"8-bit -> Memory: {memory_8bit:.2f} MB, Latency: {latency_8bit:.4f}s, Perplexity: {perplexity_8bit:.4f}")

# Load and evaluate 4-bit NF4 quantized model
print("Applying 4-bit NF4 quantization using Bitsandbytes...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model_nf4 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quantization_config, device_map="auto")
memory_nf4 = get_model_memory(model_nf4)
latency_nf4 = measure_latency(model_nf4, tokenizer, "The future of AI is", device)
perplexity_nf4 = calculate_perplexity(model_nf4, tokenizer, data, device)
print(f"4-bit NF4 -> Memory: {memory_nf4:.2f} MB, Latency: {latency_nf4:.4f}s, Perplexity: {perplexity_nf4:.4f}")

# Save models in .pth format
print("Saving models...")
torch.save(model_fp32.state_dict(), f"{QUANTIZED_MODELS_DIR}/fp32_model_weights.pth")
torch.save(model_8bit.state_dict(), f"{QUANTIZED_MODELS_DIR}/8bit_model_weights.pth")
torch.save(model_nf4.state_dict(), f"{QUANTIZED_MODELS_DIR}/nf4_model_weights.pth")
print("Models saved successfully!")

# Visualization
models = ['FP32', '8-bit', '4-bit NF4']
memory_values = [memory_fp32, memory_8bit, memory_nf4]
latency_values = [latency_fp32, latency_8bit, latency_nf4]
perplexity_values = [perplexity_fp32, perplexity_8bit, perplexity_nf4]

# Plot Memory
plt.figure(figsize=(8, 6))
plt.bar(models, memory_values, color=['blue', 'orange', 'green'])
plt.xlabel("Model Precision")
plt.ylabel("Memory (MB)")
plt.title("Memory Usage for Different Quantization Levels")
plt.show()

# Plot Latency
plt.figure(figsize=(8, 6))
plt.bar(models, latency_values, color=['blue', 'orange', 'green'])
plt.xlabel("Model Precision")
plt.ylabel("Latency (s)")
plt.title("Inference Latency for Different Quantization Levels")
plt.show()

# Plot Perplexity
plt.figure(figsize=(8, 6))
plt.bar(models, perplexity_values, color=['blue', 'orange', 'green'])
plt.xlabel("Model Precision")
plt.ylabel("Perplexity")
plt.title("Perplexity for Different Quantization Levels")
plt.show()
