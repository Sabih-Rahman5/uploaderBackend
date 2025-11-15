# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class BaseModel:
    def clear_gpu(self):
        """
        Clears GPU memory by deleting the model and tokenizer, and calling 
        torch.cuda.empty_cache() to free unused memory.
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Empty cache to release unused memory
        torch.cuda.empty_cache()
        print(f"GPU memory cleared for {self.__class__.__name__} model.")
        
        
        
        