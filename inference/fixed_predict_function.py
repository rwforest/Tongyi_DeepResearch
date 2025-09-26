from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch

def predict(model_input) -> str:
    """
    Fixed predict function that works with HuggingFace transformers
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
    model = AutoModelForCausalLM.from_pretrained("Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")

    # Only process the first item in model_input for single prediction
    item = model_input[0]
    prompt = item.get("prompt", "")
    max_length = item.get("max_length", 2048)
    temperature = item.get("temperature", float(os.environ.get('TEMPERATURE', 0.85)))
    # Note: HuggingFace doesn't support presence_penalty directly

    messages = [
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # HuggingFace generation parameters (fixed)
    generation_kwargs = {
        "max_new_tokens": max_length,
        "do_sample": True,  # Required for temperature
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
        # Remove presence_penalty - not supported by HuggingFace
    }

    # Add repetition penalty as alternative to presence_penalty
    repetition_penalty = item.get("presence_penalty", 1.1)
    if repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty

    outputs = model.generate(
        **inputs,
        **generation_kwargs
    )

    # Decode only the new tokens
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated_text


# Alternative: Even simpler version using pipeline
def predict_simple(model_input) -> str:
    """
    Simpler version using HuggingFace pipeline
    """
    item = model_input[0]
    prompt = item.get("prompt", "")
    max_length = item.get("max_length", 2048)
    temperature = item.get("temperature", 0.85)

    # Use pipeline for easier handling
    pipe = pipeline(
        "text-generation",
        model="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        tokenizer="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    messages = [{"role": "user", "content": prompt}]

    result = pipe(
        messages,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        return_full_text=False  # Only return generated text
    )

    return result[0]["generated_text"]