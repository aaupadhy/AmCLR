from tqdm import tqdm

def text_aug(text_batch, model, tokenizer):
    """
    Paraphrase a batch of texts using a T5 model fine-tuned for paraphrasing.

    Args:
        text_batch (list of str): Batch of input texts to be paraphrased.
        model: Hugging Face T5 model.
        tokenizer: Hugging Face tokenizer.

    Returns:
        list of str: List of paraphrased texts.
    """
    input_texts = [f"paraphrase: {text}" for text in tqdm(text_batch)]
    
    encoding = tokenizer(
        input_texts, 
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to("cuda")
    attention_mask = encoding["attention_mask"].to("cuda")
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=1
    )

    paraphrased_texts = [
        tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for output in outputs
    ]

    return paraphrased_texts
