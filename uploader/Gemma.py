import langchain as lc
from langchain import LLMMathChain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline


def loadModel(knowledge_base=None):
    

    model_name = "google/gemma-3-12b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Pipeline for text generation
    text_generation_pipeline = hf_pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    if(knowledge_base != None):
        loader = PyPDFLoader(knowledge_base)
        docs = loader.load()

        prompt_template = """
        You are an AI teaching assistant. Use the following context, question and student answer to provide grading and constructive feedback. 
        Ensure that the feedback includes suggestions for improvement and accuracy.
        {context}
        Question: {question}
        """
        prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,)
        
        llm_chain = prompt | llm_pipeline | StrOutputParser()

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
        chunked_docs = splitter.split_documents(docs)

        db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))
        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})

        pipeline = (
            {"context": retriever, "question": RunnablePassthrough()}
            | llm_chain
            )
    else:
        prompt_template = """
        You are an AI teaching assistant. Use the following question and student answer to provide grading and constructive feedback. 
        Ensure that the feedback includes suggestions for improvement and accuracy.
        Question: {question}
        """
        prompt = PromptTemplate( input_variables=["question"], template=prompt_template,)
        
        llm_chain = prompt | llm_pipeline | StrOutputParser()
        
        pipeline = ( {"question": RunnablePassthrough()} | llm_chain)


    return pipeline

