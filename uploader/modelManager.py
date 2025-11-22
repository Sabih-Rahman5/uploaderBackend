import re
from threading import Lock
import csv
import torch
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .DeepSeek import DeepSeek
from .Gemma import Gemma
from .Llama import LLama


def sanitize_text(text):
    return text.encode('latin-1', errors='replace').decode('latin-1')


class GPUModelManager:
    _instance = None
    _lock = Lock()
    
    class _Singleton:
        def __init__(self):
            self.model = None
            self._modelName = "None"
            #self.strictMode = False 
            self._currentState = "idle" 
            self.retriever = None
            self._progress = 0       
            self._last_error = ""
            self.retriever = None
            self.knowledgebasePath = ""
            self.assignmentPath = ""
            self.model_registry = {
                "Llama-3.2": LLama,
                "DeepSeek-r1": DeepSeek,
                "Gemma-3": Gemma
                }

        def getState(self):
            return self._currentState
        
        def getLoadedModel(self):
            return self._modelName if self._currentState == "loaded" else "None"
        
        def loadModel(self, modelName):
            if modelName not in self.model_registry:
                raise ValueError(f"Unknown model: {modelName}")

            if self.model is not None and self._modelName == modelName:
                self._currentState = "loaded"
                self._progress = 100
                return

            if self.model is not None:
                self._currentState = "clearing"
                self._progress = 5
                self.clearGpu()
                self._progress = 10

            self._currentState = "loading"
            self._progress = 15
            model_class = self.model_registry[modelName]
            self.model = model_class()
            self._progress = 30

            self.model.loadModel()
            self._modelName = modelName

            self._currentState = "loaded"
            self._progress = 100
            
            
        def setKnowledgebase(self, path):
            try:
                if self.retriever is not None:
                    print("Deleting existing retriever")
                    del self.retriever
                    self.retriever = None
                
                self.knowledgebasePath = path
                
                # 1. Loading
                pdf_loader = PyPDFLoader(file_path=path)
                docs = pdf_loader.load()
                
                # 2. Better Embeddings (Optional but recommended)
                # 'all-mpnet-base-v2' is slightly slower but much more accurate than 'all-MiniLM-L6-v2'
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                
                # 3. Improved Chunking Strategy
                # Increased size to ~1000 chars (approx 250 tokens) to keep paragraphs together
                # Increased overlap to 200 chars to ensure context flows between chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " ", ""] # Prioritize keeping paragraphs whole
                )
                
                split_docs = text_splitter.split_documents(docs)
                
                # 4. Vector Store
                db = FAISS.from_documents(split_docs, embeddings)

                # 5. Advanced Retrieval Configuration
                # k=5: We retrieve top 5 chunks to give the LLM enough context to synthesize an answer.
                # search_type="mmr": Maximal Marginal Relevance. This selects the top result, 
                # then looks for other results that are relevant but diverse, avoiding 5 identical duplicate chunks.
                retriever = db.as_retriever(
                    search_type="mmr", 
                    search_kwargs={
                        "k": 5, 
                        "fetch_k": 20,  # Fetch 20 candidates, select top 5 diverse ones
                        "lambda_mult": 0.7 # Diversity score (closer to 1 is strictly relevance, closer to 0 is max diversity)
                    }
                )
                
                self.retriever = retriever
                return True
                
            except Exception as e:
                self._last_error = str(e)
                print(f"Error setting knowledgebase: {e}")
                return False

            
        def clearGpu(self):
            self._currentState = "unloading"
            print("unloading", self._modelName)
            self.model.clear_gpu()
            del self.model
            self.model = None

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self._currentState = "empty"
    
        def extract_text_from_pdf(self):
            print(self.assignmentPath)
            reader = PdfReader(self.assignmentPath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        def extract_qa(self, text):
            # This regex looks for "question <number>:" and then "answer <same number>:".
            # It uses a backreference (\1) to ensure that the question and answer numbers match.
            pattern = re.compile(
                r"question\s*(\d+):\s*(.*?)\s*answer\s*\1:\s*(.*?)(?=question\s*\d+:|$)",
                re.IGNORECASE | re.DOTALL
            )
            # Find all matches; each match is a tuple (number, question_text, answer_text)
            matches = pattern.findall(text)
            
            # Build a dictionary mapping question numbers to a structured dictionary
            qa_dict = {}
            for num, question, answer in matches:
                qa_dict[int(num)] = {
                    'question': question.strip(),
                    'answer': answer.strip()
                }
            return qa_dict

        def runInference(self, progress_callback=None, detailed=False, strictMode=False):
            try:
                pdf_text = self.extract_text_from_pdf()
                qa_pairs = self.extract_qa(pdf_text)
            
                pdf = FPDF()
                pdf.add_page()
                
                # --- ROBUST PAGE HANDLING CONFIGURATION ---
                # 1. Enable automatic page breaks. 
                # margin=15 leaves 15mm at the bottom.
                pdf.set_auto_page_break(auto=True, margin=15) 
                
                pdf.set_font("Arial", size=12)
                
                # Helper function to check space before adding a new block header
                # This prevents a header (like "Question 1") from appearing alone at the bottom of a page
                def check_space_for_header(pdf_obj, required_space=40):
                    # Get current Y position
                    current_y = pdf_obj.get_y()
                    # Get page height (usually 297mm for A4) minus margin
                    page_height = pdf_obj.h - 20 
                    
                    if current_y + required_space > page_height:
                        pdf_obj.add_page()

                total = len(qa_pairs)        
                scores = []
                scoreCount = 0
                
                for i, number in enumerate(sorted(qa_pairs, key=int)):
                    print(f"Processing Question {number}...")
                    
                    qa = qa_pairs[number]
                    question = qa["question"]
                    answer = qa["answer"]
                    
                    # --- ENSURE LOGIC ---
                    # Before starting a new Q/A block, check if we have enough space
                    # to at least fit the headers and some text.
                    check_space_for_header(pdf)

                    # Add Question Heading
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Question {number}:", ln=True)
                    
                    # Add Question Text
                    pdf.set_font("Arial", "", 12)
                    # multi_cell will now automatically page break because of set_auto_page_break above
                    pdf.multi_cell(0, 10, sanitize_text(question))
                    pdf.ln(2)
                    
                    # Check space before Answer section to keep it somewhat together
                    check_space_for_header(pdf, required_space=30)

                    # Add Answer Heading
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Answer {number}:", ln=True)
                    
                    # Add Answer Text
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, sanitize_text(answer))
                    pdf.ln(2)
                    
                    # Generate Model Feedback
                    feedback = self.model.runInference(question, answer, self.retriever, detailed)
                    
                    # Extract Score
                    match = re.search(r'(\d+)%', feedback)
                    if match:
                        score = match.group(1)
                        print(f"Accuracy score: {score}%")
                        scores.append(int(score))
                        scoreCount += 1
                    else:
                        print("No score found in the output.")
                    
                    # Check space before Feedback section
                    check_space_for_header(pdf, required_space=30)

                    # Add Feedback Heading
                    pdf.set_font("Arial", "B", 12)
                    if detailed:
                        pdf.cell(0, 10, f"Results: {number}:", ln=True)
                    else:
                        pdf.cell(0, 10, f"Feedback {number}:", ln=True)
                    
                    # Add Feedback Text
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, sanitize_text(feedback))
                    pdf.ln(5)  # Add a gap before next question-answer pair
                        
                    if progress_callback is not None:
                        progress_callback((i + 1) / total)
                        
                total_score = sum(scores) / scoreCount if scoreCount > 0 else 0        
                
                # Save CSV
                with open('scores.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Question Number', 'Accuracy (%)'])
                    for number, score in zip(sorted(qa_pairs.keys()), scores):
                        writer.writerow([number, score])
                    writer.writerow(['Total Accuracy', f'{total_score:.2f}%'])
                
                pdf.output("output.pdf")
                return True
            
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc() # Helpful to see where it failed
                return False





    @classmethod
    def getInstance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls._Singleton()
        return cls._instance
