import pymongo
import os
from dotenv import load_dotenv
import certifi
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
import pandas as pd
from openai import OpenAI as OpenAIClient
from openai import AzureOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate


# Load variables 
load_dotenv()
# clientOpenAI = OpenAIClient(api_key=os.environ['OPENAI_API_KEY'])
# Hide the Streamlit menu and footer
hide_menu_style = """
        <style>
        MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

clientAzureOpenAI = AzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), 
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('OPENAI_API_VERSION'))

os.environ['SSL_CERT_FILE'] = certifi.where()
MONGODB_CONNECTION_STRING = os.environ.get('MONGODB_CONNECTION_STRING')

# Function to connect to MongoDB and return a collection
def connect_mongodb(col):
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
# change to your database name    
    db = client["demo_hc_hospital"]
    return db[col]

# change the model deployment as per your deployed embedding model
def get_embedding(text, model="leafy-ada-002-model"):
    text = text.replace("\n", " ")
    # print(text)
    response = clientAzureOpenAI.embeddings.create(input=[text], model=model)
    embedding = response['choices'][0]['embedding'] if isinstance(response, dict) else response.data[0].embedding
    return embedding

# Function to process PDFs in a directory
def process_pdf_directory(directory_path):
    data = []
    files = os.listdir(directory_path)
    total_files = len(files)
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    for idx, filename in enumerate(files):
        if filename.endswith(".pdf"):
            status_placeholder.text(f"Processing file {filename} ({idx+1}/{total_files})...")
            pdf_path = os.path.join(directory_path, filename)
            with open(pdf_path, "rb") as f:
                pdf_reader = PdfReader(f)
                for page_number, page in enumerate(pdf_reader.pages, start=1):
                    data.append({
                        "text": page.extract_text(),
                        "filename": filename,
                        "page_number": page_number
                    })
            progress_bar.progress((idx+1)/total_files)

    progress_bar.progress(100)
    status_placeholder.text("All files processed!")
    return data

# Function to store text embeddings in MongoDB
def store_text_embeddings(data):
# change to your MongoDB collection name    
    collection = connect_mongodb(col="RAGpatientGuides")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    total_chunks = len(data)
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    for idx, entry in enumerate(data):
        chunks = text_splitter.split_text(entry['text'])
        for chunk in chunks:
            embedding = get_embedding(chunk)
            document = {
                "text_chunk": chunk,
                "vector_embedding": embedding,
                "source": {
                    "filename": entry['filename'],
                    "page_number": entry['page_number']
                }
            }
            collection.insert_one(document)
        
        progress_bar.progress((idx+1)/total_chunks)
        status_placeholder.text(f"Storing chunks from {entry['filename']} ({idx+1}/{total_chunks}) in MongoDB...")

    status_placeholder.text("All text chunks stored in MongoDB!")

# Function to find similar documents based on embeddings
def find_similar_documents(embedding, k):
    print("Searching for similar documents in RAGpatientGuides...")
    collection = connect_mongodb(col="RAGpatientGuides")
    documents = list(collection.aggregate([{
        "$vectorSearch": {
            "index": "default",
            "path": "vector_embedding",
            "queryVector": embedding, 
            "numCandidates": 200, 
            "limit": k
        }
    }]))

    print(f"Found {len(documents)} similar documents in RAGpatientGuides.")
    
    return documents

def main():
    st.header("Load Documents")
    dir_path = st.text_input("Enter the directory path containing PDFs:")
    # print("dir_path", dir_path)
    # dir_path = None
    if dir_path:
        combined_text = process_pdf_directory(dir_path)
        if combined_text:
            store_text_embeddings(combined_text)
            st.success("Documents loaded and embeddings stored!")
        else:
            st.warning("No text was extracted from the PDFs. Check the directory and files.")


if __name__ == "__main__":
    main()