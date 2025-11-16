import gc
import torch

class BaseModel:
    def __init__(self):
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def loadModel(self):
        raise NotImplementedError

    def _print_cuda_stats(self, tag=""):
        try:
            allocated = torch.cuda.memory_allocated()
            reserved  = torch.cuda.memory_reserved()
            print(f"[CUDA stats {tag}] allocated={allocated:,} reserved={reserved:,}")
        except Exception as e:
            print(f"[CUDA stats {tag}] error: {e}")

    def clear_gpu(self):
        """
        Robust cleanup: break references in pipeline, move large tensors to CPU, delete,
        run GC, then empty cache and sync.
        """
        print(f"Clearing GPU for {self.__class__.__name__} ({self.model_name})...")
        self._print_cuda_stats("before")

        # 1) Try to clear pipeline internals first (pipeline often holds model/tokenizer)
        try:
            if getattr(self, "pipeline", None) is not None:
                try:
                    # If it's a transformers.pipeline.Pipeline, clear inside references
                    # set model/tokenizer attrs to None to break references
                    if hasattr(self.pipeline, "model"):
                        self.pipeline.model = None
                    if hasattr(self.pipeline, "tokenizer"):
                        self.pipeline.tokenizer = None
                except Exception:
                    pass
                # delete the pipeline object
                del self.pipeline
                self.pipeline = None
                print("  - Pipeline cleared.")
        except Exception as e:
            print("  - Error clearing pipeline:", e)

        # 2) Move model to CPU (helps if device_map placed shards)
        try:
            if getattr(self, "model", None) is not None:
                try:
                    # attempt to move parameters to CPU to release GPU memory
                    self.model.to("cpu")
                except Exception:
                    # some model wrappers may not support .to() — ignore
                    pass
                del self.model
                self.model = None
                print("  - Model deleted.")
        except Exception as e:
            print("  - Error deleting model:", e)

        # 3) Delete tokenizer
        try:
            if getattr(self, "tokenizer", None) is not None:
                del self.tokenizer
                self.tokenizer = None
                print("  - Tokenizer deleted.")
        except Exception as e:
            print("  - Error deleting tokenizer:", e)

        # 4) Force GC and empty CUDA cache & sync
        gc.collect()
        # PyTorch allocator helpers
        try:
            torch.cuda.ipc_collect()  # may be no-op on some setups
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        except Exception:
            pass

        self._print_cuda_stats("after")
        print("GPU clear attempt finished. If memory still shows as used, check for other references or external processes (nvidia-smi).")
