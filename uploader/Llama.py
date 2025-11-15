# import langchain as lc
# from langchain import LLMMathChain
# from langchain.chains import RetrievalQA
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# # from langchain.schema import Document
# # import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import pipeline as hf_pipeline
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_huggingface import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .BaseModel import BaseModel

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

class LLama(BaseModel):
    def __init__(self):
        # Initialize model and tokenizer (if needed to be shared across methods)
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def loadModel(self):
        llama_pipe = pipeline(
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
        return llama_pipe
