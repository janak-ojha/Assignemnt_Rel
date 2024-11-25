import requests
from bs4 import BeautifulSoup

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract all text from paragraphs or other desired HTML elements
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    
    return text

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the model to convert text to embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def store_in_faiss(text):
    # Split text into chunks
    chunks = text.split('\n\n')  # Example: splitting by paragraphs
    
    # Convert text chunks to embeddings
    embeddings = model.encode(chunks)
    
    # Create a Faiss index
    dimension = embeddings.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(np.array(embeddings))  # Add embeddings to the index
    
    return index, chunks


def retrieve_from_faiss(query, faiss_index, text_chunks):
    # Convert the query into embeddings
    query_embedding = model.encode([query])
    
    # Perform a search in the Faiss index
    D, I = faiss_index.search(np.array(query_embedding), k=3)  # Get top 3 closest chunks
    
    # Retrieve the relevant text chunks
    relevant_chunks = [text_chunks[i] for i in I[0]]
    
    # Combine the chunks into context
    context = ' '.join(relevant_chunks)
    return context
from langchain_huggingface import HuggingFaceEndpoint

# Create the HuggingFaceEndpoint instance (as in your original code)
sec_key = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    token=sec_key
)

# Extract data from the website and store in Faiss
url = "https://botpenguin.com/"
text = extract_text_from_url(url)
faiss_index, text_chunks = store_in_faiss(text)

# Define a function to get context from Faiss and call the model
def ask_question_with_faiss(query):
    context = retrieve_from_faiss(query, faiss_index, text_chunks)
    
    # Add the context to the prompt
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    # Get the model response
    response = llm.invoke(prompt)
    return response

# Ask a question
response = ask_question_with_faiss("What  are services provided bot penguin")
print(response)
