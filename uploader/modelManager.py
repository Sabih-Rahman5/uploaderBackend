from .DeepSeek import DeepSeek
from .Llama import Llama
from .Gemma import Gemma
from threading import Lock  
import torch

from pypdf import PdfReader
from fpdf import FPDF
import re

def sanitize_text(text):
    return text.encode('latin-1', errors='replace').decode('latin-1')


class GPUModelManager:
    _instance = None
    _lock = Lock()
    
    class _Singleton:
        def __init__(self):
            self._modelName = ""
            self._currentState = "empty"
            self.model = None
            self.knowledge_base = None
            self.assignment = None

        def getState(self):
            return self._currentState
        
        def getLoadedModel(self):
            return self._modelName if self._currentState == "loaded" else None
        
        def loadModel(self, modelname):        
            self._currentState = "loading"
            self._modelName = modelname
            
            if(modelname == "DeepSeek-r1"):
                self.model = DeepSeek.loadModel(self.knowledge_base)
                self._currentState = "loaded"                
            if(modelname == "Gemma-3"):
                self.model = Gemma.loadModel(self.knowledge_base)
                self._currentState = "loaded"
                
            if(modelname == "Llama-3.2"):
                self.model = Llama.loadModel(self.knowledge_base)
                self._currentState = "loaded"
        
        def clearGpu(self):
            if self._currentState == "loaded":
                self._currentState = "unloading"
                print("unloading model: " + str(self._modelName))
                
                del self.model
                self.model = None

                torch.cuda.empty_cache()
                # torch.cuda.ipc_collect()
                # torch.cuda.reset_peak_memory_stats()
                # torch.cuda.synchronize()
                self._currentState = "empty"
    
        def extract_text_from_pdf(self):
            print(self.assignment)
            reader = PdfReader(self.assignment)
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
            
        # Example function to create the PDF

        def runInference(self, progress_callback=None):
            try:
                pdf_text = self.extract_text_from_pdf()
                qa_pairs = self.extract_qa(pdf_text)
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # for number in sorted(qa_pairs):
                #     qa = qa_pairs[number]
                #     print(f"Question {number}: {qa['question']}")
                #     print(f"Answer {number}: {qa['answer']}\n")
                
                total = len(qa_pairs)
                for i, number in enumerate(sorted(qa_pairs, key=int)):  # assuming keys are numeric strings
                    
                    print(f"Processing Question {number}...")
                    
                    qa = qa_pairs[number]
                    question = qa["question"]
                    answer = qa["answer"]
                    
                    
                    # Add Question Heading
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Question {number}:", ln=True)
                    # Add Question Text
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, sanitize_text(question))
                    pdf.ln(2)
                    
                    # Add Answer Heading
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Answer {number}:", ln=True)
                    # Add Answer Text
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, sanitize_text(answer))
                    pdf.ln(2)
                    
                    feedback = self.model.invoke(str(question + "\n" + answer))
                    
                    
                    # Add Feedback Heading
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Feedback {number}:", ln=True)
                    # Add Feedback Text
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, 10, sanitize_text(feedback))
                    pdf.ln(5)  # Add a gap before next question-answer pair
                        
                        
                    if progress_callback is not None:
                        progress_callback((i + 1) / total)

                pdf.output("output.pdf")
                return True
            
            except Exception as e:
                print(f"Error: {e}")
                return False
            # prompt = ""
            # feedback = self.model.invoke(str(prompt))
            # print(feedback)

    @classmethod
    def getInstance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls._Singleton()
        return cls._instance
