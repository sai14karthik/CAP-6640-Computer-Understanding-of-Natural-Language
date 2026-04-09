"""
Load open-source LLMs via Hugging Face Transformers for inference.
"""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODELS, GENERATION_CONFIG


def load_model_and_tokenizer(
    model_key: str,
    use_8bit: bool = True,
    device_map: Optional[str] = "auto",
    force_cpu: bool = False,
) -> tuple:
    """
    Load model and tokenizer by config key.
    use_8bit: use 8-bit quantization to reduce memory (GPU only).
    force_cpu: load on CPU (no GPU); use with small models like gpt2, distilgpt2.
    """
    model_id = MODELS.get(model_key)
    if not model_id:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODELS.keys())}")

    on_cpu = force_cpu or not torch.cuda.is_available()
    kwargs = {}
    # 8-bit loaders (bitsandbytes) are for large checkpoints; small LMs often fail or gain nothing.
    small_lm = model_key in ("gpt2", "distilgpt2")
    if on_cpu:
        kwargs["device_map"] = None
        kwargs["torch_dtype"] = torch.float32
    else:
        if use_8bit and not small_lm:
            kwargs["load_in_8bit"] = True
            kwargs["device_map"] = device_map
        else:
            kwargs["device_map"] = device_map
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        **kwargs,
    )
    if on_cpu or (hasattr(model, "device") and model.device.type == "cpu"):
        model = model.to("cpu")
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """Generate a single answer from the model."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_cfg["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(**inputs, **gen_cfg)
    # Decode only the generated part
    generated = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text
