from transformers import T5ForConditionalGeneration, T5Tokenizer

def text_aug(text, model_name="t5-small", max_length=100):
    """
    Paraphrase a given text using a lightweight model from Hugging Face.
    
    Args:
        text (str): Input text to be paraphrased.
        model_name (str): Name of the Hugging Face model to use (default: "t5-small").
        max_length (int): Maximum length of the paraphrased output.
    
    Returns:
        str: Paraphrased text.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    input_text = f"paraphrase: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    
    outputs = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
    
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text