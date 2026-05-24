from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "vennify/t5-base-grammar-correction"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Move model to GPU
model = model.to(device)

# User input
input_text = input("Enter a sentence: ")

input_ids = tokenizer.encode(
    "grammar: " + input_text,
    return_tensors="pt",
    max_length=128,
    truncation=True
).to(device)

outputs = model.generate(input_ids, max_length=128)

corrected_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nCorrected Sentence:", corrected_text)