# rag_engine.py

import os
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from knowledge_base import arogya_vidya_data

# --- CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- GLOBAL VARIABLES ---
# We will store our processed data here to avoid re-calculating
text_chunks = []
vector_store = None

def setup_rag_engine():
    """
    Processes the knowledge base text and sets up the FAISS vector store.
    This function should be run once when the application starts.
    """
    global text_chunks, vector_store

    if text_chunks and vector_store:
        print("RAG engine already configured.")
        return

    print("Setting up RAG engine...")
    # 1. Chunking: Split the text by paragraphs
    text_chunks = [paragraph.strip() for paragraph in arogya_vidya_data.split('\n\n') if paragraph.strip()]
    if not text_chunks:
        raise ValueError("No text chunks found. Check the knowledge base.")

    # 2. Embedding: Convert text chunks to vectors
    print(f"Embedding {len(text_chunks)} text chunks...")
    model = 'models/embedding-001'
    # Use a try-except block to handle potential API errors
    try:
        embeddings = genai.embed_content(model=model,
                                         content=text_chunks,
                                         task_type="retrieval_document")['embedding']
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        return

    # 3. Vector Store: Create and populate the FAISS index
    print("Creating FAISS vector store...")
    embedding_dim = len(embeddings[0])
    vector_store = faiss.IndexFlatL2(embedding_dim)
    vector_store.add(np.array(embeddings))
    print("RAG engine setup complete!")

def get_response(user_query):
    """
    Gets a response from the LLM based on the user's query.
    """
    if not vector_store:
        raise RuntimeError("Vector store is not initialized. Run setup_rag_engine() first.")

    print(f"Received query: {user_query}")
    # 1. Embed the user's query
    model = 'models/embedding-001'
    query_embedding = genai.embed_content(model=model,
                                          content=user_query,
                                          task_type="retrieval_query")['embedding']

    # 2. Search the vector store for the most relevant chunks
    k = 6 # Number of relevant chunks to retrieve, 
    distances, indices = vector_store.search(np.array([query_embedding]), k)

    # 3. Retrieve the actual text chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n---\n".join(relevant_chunks)
    print(f"Retrieved context: \n{context}")

    # 4. Construct the prompt
    # This is the NEW and IMPROVED prompt
    system_prompt = """
    You are a compassionate virtual assistant for Arogya Vidya™, an online medical consultation service by Dr. Vidya Kollu. Your purpose is to help users understand the services offered and guide them to book an appointment.

    RULES:
    - Answer questions strictly based on the provided CONTEXT. Do not use any external knowledge.
    - If the context doesn't contain the answer, say "I'm sorry, I don't have information on that topic. I can help with questions about Dr. Kollu's services."
    - **WHEN ASKED A QUESTION THAT REQUIRES COMPARISON (like finding the 'most' or 'least' expensive), CAREFULLY EXAMINE ALL RELEVANT PRICES IN THE CONTEXT BEFORE DETERMINING THE ANSWER.**
    - **DO NOT give medical advice.** If the user asks for a diagnosis, interprets symptoms, or asks for medical advice, you MUST respond with:
    'As an AI assistant, I cannot provide any medical advice. It is very important to discuss your personal health situation directly with a qualified doctor. Dr. Vidya Kollu specializes in providing this kind of expert guidance. You can book an appointment on the arogyavidya.com website.'
    - Be empathetic, clear, and supportive in all your responses.
    """
    
    final_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nUSER'S QUESTION:\n{user_query}\n\nYOUR ANSWER:"

    # 5. Generate the response
    print("Generating response from Gemini (gemini-2.5-flash)...")
    llm = genai.GenerativeModel('gemini-2.5-flash')
    response = llm.generate_content(final_prompt)
    
    return response.text