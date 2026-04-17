import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Module-level variables to hold the model and tokenizer in memory
_model = None
_tokenizer = None
_model_id = "google/gemma-2b-it"

def _initialize_model():
    """Private function to load the model and tokenizer into VRAM safely."""
    global _model, _tokenizer
    
    # Safety check to prevent double-loading
    if _model is not None and _tokenizer is not None:
        return
        
    print(f"Loading {_model_id} into memory...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,   
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"              
    )

    _tokenizer = AutoTokenizer.from_pretrained(_model_id)

    _model = AutoModelForCausalLM.from_pretrained(
        _model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,             
        low_cpu_mem_usage=True                 
    )

    _model.eval()
    print("Model loaded successfully on:", _model.device)

def llm_call(prompt, system=None):
    """Public function to query the Gemma model."""
    global _model, _tokenizer
    
    # Lazy loading: Only initialize the model the first time this function is called
    if _model is None or _tokenizer is None:
        _initialize_model()
        
    full_prompt = ""
    if system:
        full_prompt += system + "\n\n"
    full_prompt += prompt

    # Tokenize and move to GPU
    inputs = _tokenizer(full_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=60,      
            temperature=0,         
            do_sample=False,
            use_cache=True          
        )

    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the output by removing the echoed prompt
    response = response.replace(full_prompt, "").strip()

    return response