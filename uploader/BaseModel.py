from dataclasses import replace
import gc
import torch
from ddgs import DDGS
from uploader.utils import criticValidatorEvaluator


critic_prompt = """<INSTRUCTION>
You are a strict answer critic. Your *sole task* is to identify statements in the <ANSWER> that are *factually contradicted* by the <CONTEXT>.

RULES:
- Use ONLY the <CONTEXT>. Do not use any outside knowledge.
- A "contradiction" is when the <ANSWER> states something that is proven false by the <CONTEXT>.
- **Example:** If <CONTEXT> says "The heart pumps blood" and <ANSWER> says "The lungs pump blood", this is a contradiction.
- Respond ONLY with a numbered list of the contradictions.
- Do NOT comment on what is missing from the context.
- Do NOT add any preamble or commentary.
- If no contradictions are found, reply exactly: "None"
</INSTRUCTION>

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>

<ANSWER>
{answer}
</ANSWER>

<RESPONSE FORMAT>
1. <The reason the answer is contradicted by the context>
</RESPONSE FORMAT>

<RESPONSE>
"""

sage_prompt = """<INSTRUCTION>
You are a strict answer verifier. Your *sole task* is to identify statements in the <ANSWER> that are *factually supported* by the <CONTEXT>.

RULES:
- Use ONLY the <CONTEXT>. Do not use any outside knowledge.
- A "supported" statement is one that is explicitly confirmed or logically equivalent to information in the <CONTEXT>.
- If a statement in the <ANSWER> is correct but not mentioned in the <CONTEXT>, IGNORE it (do not list it as correct or incorrect).
- If a statement is partially correct, include only the part that is fully supported by the <CONTEXT>.
- Respond ONLY with a numbered list of the correct statements from the <ANSWER>.
- Do NOT include commentary, speculation, or missing details.
- If no correct statements are found, reply exactly: "None"
</INSTRUCTION>

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>

<ANSWER>
{answer}
</ANSWER>

<RESPONSE FORMAT>
1. <The statement from the answer that is supported by the context>
</RESPONSE FORMAT>

<RESPONSE>
"""

score_prompt = """<INSTRUCTION>
You are an **AI evaluator providing feedback**. Your *primary task* is to **provide constructive feedback to a student** and then assign a percentage score (0–100%) representing their answer's accuracy, based on the provided lists of <ACCURACIES> and <INACCURACIES>.
</INSTRUCTION>

<RULES>
- **First, provide constructive feedback (1-2 sentences). This feedback should be encouraging, acknowledge what was correct (from <ACCURACIES>), and then clearly point out the main areas for improvement (from <INACCURACIES>).**
- **The feedback's purpose is to *help the student learn*, not just to justify the score.**
- Consider both the number and severity of inaccuracies versus accuracies when determining the score.
- The score must reflect *overall factual accuracy*.
- Use the following scale as guidance:
  - 100% = Perfectly accurate, no inaccuracies.
  - 80–99% = Mostly accurate, only minor inaccuracies.
  - 60–79% = Mixed accuracy; several errors but generally correct.
  - 40–59% = More incorrect than correct.
  - 20–39% = Largely inaccurate, few correct statements.
  - 0–19% = Almost entirely inaccurate.
- After the feedback, give the final percentage score clearly on a new line.
- Respond only in the specified format, without commentary or extra text.
</RULES>

<ACCURACIES>
{accuracies}
</ACCURACIES>

<INACCURACIES>
{inaccuracies}
</INACCURACIES>

<RESPONSE FORMAT>
Feedback: <brief, constructive feedback summarizing strengths and areas for improvement>
Final accuracy score: <percentage>%
</RESPONSE FORMAT>

<RESPONSE>
"""

relevance_prompt = """<INSTRUCTION>
You are a highly-strict relevance-checking AI. Your *sole task* is to determine if the <QUESTION> can be answered *exclusively* using the information found within the <RETRIEVED CONTEXT>.

You must not answer the question or use any external knowledge.
</INSTRUCTION>

<RULES>
1.  **If** the <RETRIEVED CONTEXT> contains information that **directly answers** the <QUESTION> (even if it's not a comprehensive, exhaustive answer), you must respond with **true**.
2.  **If** the <RETRIEVED CONTEXT> is empty, you must respond with **false**.
3.  **If** the <RETRIEVED CONTEXT> is irrelevant to the <QUESTION>, you must respond with **false**.
4.  **If** the <RETRIEVED CONTEXT> mentions the topic but *does not contain the specific information* needed to answer the <QUESTION> (e.g., context mentions the heart but not its function), you must respond with **false**.
5.  Your response must be *only* the word 'true' or 'false' in lowercase, with no other text, explanation, or punctuation.
</RULES>

<QUESTION>
{question}
</QUESTION>

<RETRIEVED CONTEXT>
{retrieved_context}
</RETRIEVED CONTEXT>

<RESPONSE>
"""



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

    def isContextRelevant(self, question, context):
        if self.pipeline is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before running inference.")
        # Fill prompt
        prompt = relevance_prompt.format(
            question=question,
            retrieved_context=context
        )
        
        output = self.pipeline(prompt, eos_token_id=self.tokenizer.eos_token_id)[0]["generated_text"]
        if output.strip().lower() == "true":
            return True
        else:
            return False

    def getWebContext(self, query):
        paragraphs = []
        with DDGS() as ddgs:
            results = ddgs.text(query, safesearch="off", max_results=10)
            for r in results:
                title = r.get("title", "")
                snippet = r.get("body", "")   # this is the paragraph-like text
                url = r.get("href", "")
        
                if snippet:
                    paragraphs.append(snippet)

        # Combine into a single context block
        context = "\n\n".join(paragraphs)
        return context


    def criticValidatorEvaluator(self, question, answer, context):
        prompt = critic_prompt.format(
            question=question,
            context=context,
            answer=answer)

        output = self.pipeline(prompt, eos_token_id=self.tokenizer.eos_token_id)[0]["generated_text"]
        return output
    
    def sageValidatorEvaluator(self, question, answer, context):
        prompt = sage_prompt.format(
            question=question,
            context=context,
            answer=answer)
        output = self.pipeline(prompt, eos_token_id=self.tokenizer.eos_token_id)[0]["generated_text"]
        return output

    def sorcerer(self, accuracies, inaccuracies):
 
        # Fill prompt
        prompt = promptTemplate.format(
            accuracies=accuracies,
            inaccuracies=inaccuracies
        )
        # print (prompt)
        output = llama_pipe(prompt, eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]
        return output


    def runInference(self, question, answer, retriever):    
        if self.pipeline is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before running inference.")
        
        context = ""
        if retriever is not None:    
            retrieved_docs = retriever.invoke(question)
            print(retrieved_docs)
            context = "\n".join([d.page_content for d in retrieved_docs])
    
        if not self.isContextRelevant(question, context):
            print("Context is not relevant to the question.")
            context = self.getWebContext(question)


        criticResponse = self.criticValidatorEvaluator(question, answer, critic_prompt, model, tokenizer).split("<RESPONSE>", 1)[-1].replace("</RESPONSE>", "")
        sageResponse = self.sageValidatorEvaluator(question, answer, sage_prompt, model, tokenizer).split("<RESPONSE>", 1)[-1].replace("</RESPONSE>", "")
        sorcererResponse = self.sorcerer(sageResponse, criticResponse).split("<RESPONSE>", 1)[-1].replace("</RESPONSE>", "")


        output = sorcererResponse
        return output