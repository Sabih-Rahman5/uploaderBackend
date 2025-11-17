from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .BaseModel import BaseModel

class DeepSeek(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    
    def loadModel(self):
        # Prevent re-loading if already loaded
        if self.pipeline:
            print(f"{self.model_name} is already loaded.")
            return

        print(f"Loading {self.model_name} model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Creating pipeline for {self.model_name}...")
        # Create and STORE the pipeline as an attribute
        self.pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=200,
            temperature=0.5,
            top_p=0.5
        )
        print(f"{self.model_name} loaded successfully.")