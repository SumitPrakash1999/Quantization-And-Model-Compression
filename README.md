# Quantization-And-Model-Compression
Implementing Quantization using Selective Components, BitsAndBytes, NF4 Quantization

## Instructions to Execute the Code
Clone the repository and ensure all necessary files and dependencies are installed:
```
bash
pip install -U transformers datasets bitsandbytes matplotlib tqdm
```
Run the code provided for quantization and evaluation:
```
python
python main.py
```
Save and load model weights using the following code snippets:

- Saving Model Weights
```
python
import torch

# Define paths
ORIGINAL_MODEL_PATH = "./original_model_weights.pth"
QUANTIZED_8BIT_PATH = "./8bit_model_weights.pth"
QUANTIZED_4BIT_PATH = "./4bit_model_weights.pth"

# Save weights
torch.save(original_model.state_dict(), ORIGINAL_MODEL_PATH)
torch.save(quantized_8bit_model.state_dict(), QUANTIZED_8BIT_PATH)
torch.save(quantized_4bit_model.state_dict(), QUANTIZED_4BIT_PATH)
```

- Loading Model Weights
```
python
import torch
from transformers import AutoModelForCausalLM

# Load model weights
original_model.load_state_dict(torch.load("./original_model_weights.pth"))
quantized_8bit_model.load_state_dict(torch.load("./8bit_model_weights.pth"))
quantized_4bit_model.load_state_dict(torch.load("./4bit_model_weights.pth"))
```

### Model Weight Links
- Original Model Weights: https://drive.google.com/file/d/15o53uOcILzufyvQoBB9uXoJ-mKL0Zlof/view?usp=sharing
- Quantized Model Weights: https://drive.google.com/file/d/103R52OlCERBWar8oQohAt-SS4EBB3ZVt/view?usp=sharing
- 8-bit Quantized Model Weights: https://drive.google.com/file/d/1UIq1MBfjnPQNyHyf6Vp2gvPwiPncM8hl/view?usp=sharing
- 4-bit Quantized Model Weights: https://drive.google.com/file/d/1Dd-ilHwnIE4bp2f7RUp9scZkK9Jhwynu/view?usp=sharing
