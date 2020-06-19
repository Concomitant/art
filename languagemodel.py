## This is just a wrapper around hugging face's transformer library. Which is already a wrapper around
## pytorch/tensorflow. 


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LanguageModel:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Auto pad with EOS because we don't have time for that nonsense.
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id)
    
    def generate_from_seed(self, seed_text, maxlen=20):
        # We will need to figure something out for where users
        # erase their own text, if we want it to start erasing at the same time
        # might be easiest to turn off early stopping (option below)
        
        torch.manual_seed(0)
        
        # Temperature scaling, k sampling and probability mass sampling...
        # Note: Maybe lower k and more sentences?
        
        input_ids = self.tokenizer.encode(seed_text, return_tensors='pt')
        
        sample_outputs = self.model.generate(
            input_ids,
            do_sample=True, 
            max_length=50, 
            top_k=50, 
            top_p=0.95, 
            num_return_sequences=3)
        

        return [self.tokenizer.decode(sample_output, skip_special_tokens=True) for sample_output in sample_outputs]
