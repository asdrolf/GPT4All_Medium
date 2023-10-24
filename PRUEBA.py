import os
import torch
import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.faiss import FAISS

# ---- HuggingFace LLM Wrapper ----
class HuggingFaceLLM:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            device_map="auto",
        )
        
    def generate(self, prompt):
        generation = self.generator(
            prompt,
            do_sample=True,
            top_k=10,
            max_new_tokens=150,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return generation[0]['generated_text']

def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return search_index

# ---- Initialize Components ----

# Paths
llama_path = './models/ggml-model-q4_0.bin' 

# FAISS index and embeddings
embeddings = LlamaCppEmbeddings(model_path=llama_path)

# Hugging Face model
huggingface_llm = HuggingFaceLLM(model_id="projecte-aina/aguila-7b")

# Load or create the database
try:
    index = FAISS.load_local("my_faiss_index", embeddings)
except:
    # If not found, index the documents
    pdf_folder_path = './docs'
    doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
    
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[0]))
    docs = loader.load()
    chunks = split_chunks(docs)
    index = create_index(chunks)

    for i in range(1, len(doc_list)):
        loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
        docs = loader.load()
        chunks = split_chunks(docs)
        new_index = create_index(chunks)
        index.merge_from(new_index)
    
    index.save_local("my_faiss_index")

# Template for the prompt
template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: """

# Hardcoded question
question = "What is a PLC and what is the difference with a PC?"
context = "\n".join([doc.page_content for doc in index.similarity_search(question, k=3)])
prompt = template.format(context=context, question=question)
response = huggingface_llm.generate(prompt)

print(f"Result: {response}")
