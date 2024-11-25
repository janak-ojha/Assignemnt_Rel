import streamlit as st
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Function to extract text from a given URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract all text from paragraphs or other desired HTML elements
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    
    return text

# Initialize the model to convert text to embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to store text in Faiss index
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

# Function to retrieve relevant text from Faiss index based on a query
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

# HuggingFace API setup
sec_key = os.getenv("HF_API_KEY")
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    token=sec_key
)

# Extract data from the website and store it in Faiss index
url = "https://botpenguin.com/"
text = extract_text_from_url(url)
faiss_index, text_chunks = store_in_faiss(text)

# Streamlit app UI
st.title("Bot Penguin Services Query")
st.write("""
    This application lets you ask questions about the services provided by Bot Penguin.
    Simply type your question, and the model will provide the relevant answer based on the content from the website.
""")

# Input box for user query
query = st.text_input("Enter your question about Bot Penguin's services:")

# If the user submits a query
if query:
    # Retrieve context from Faiss
    context = retrieve_from_faiss(query, faiss_index, text_chunks)
    
    # Add the context to the prompt
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    # Get the model response
    response = llm.invoke(prompt)
    
    # Display the response
    st.write("Answer:")
    st.write(response)

