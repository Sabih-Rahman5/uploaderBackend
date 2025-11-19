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
            self._currentState = "idle" 
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
                self.knowledgebasePath = path
                loader = PyPDFLoader(path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
                split_docs = text_splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                db = FAISS.from_documents(split_docs, embeddings)
                self.retriever = db.as_retriever(search_kwargs={"k": 1})
                return True
            except Exception as e:
                self._last_error = str(e)
                print(f"setKnowledgebase error: {e}")
                self.retriever = None
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

        def runInference(self, progress_callback=None, detailed=False):
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
                scores = []
                scoreCount = 0
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
                    
                    feedback = self.model.runInference(question, answer, self.retriever, detailed)
                    
                    
                    match = re.search(r'(\d+)%', feedback)
                    
                    if match:
                        score = match.group(1)  # Extracts the number part of the score
                        print(f"Accuracy score: {score}%")
                        scores.append(int(score))
                        scoreCount += 1
                    else:
                        print("No score found in the output.")
                    
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
                
                # Save the individual scores to a CSV file
                with open('scores.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Question Number', 'Accuracy (%)'])  # Header row
                    for number, score in zip(sorted(qa_pairs.keys()), scores):
                        writer.writerow([number, score])  # Write each question's score
                        
                    writer.writerow(['Total Accuracy', f'{total_score:.2f}%'])
                        
                      
                        
                
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
