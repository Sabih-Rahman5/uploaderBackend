# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class BaseModel:
    def __init__(self):
        """
        Initialize attributes to None. 
        No heavy models are loaded here.
        """
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None  # We will now store the pipeline

    def loadModel(self):
        """
        This method will be implemented by subclasses to load the
        model, tokenizer, and create the pipeline.
        """
        raise NotImplementedError

    def clear_gpu(self):
        """
        Clears GPU memory by deleting the pipeline, model, and tokenizer
        in the correct order.
        """
        print(f"Clearing GPU for {self.__class__.__name__}...")
        
        # Delete the pipeline FIRST, as it holds references
        # to the model and tokenizer.
        if hasattr(self, 'pipeline'):
            del self.pipeline
            self.pipeline = None
            print("  - Pipeline deleted.")

        if hasattr(self, 'model'):
            del self.model
            self.model = None
            print("  - Model deleted.")

        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
            print("  - Tokenizer deleted.")

        
        # Empty cache to release unused memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f"GPU memory cleared and cache emptied for {self.model_name}.")