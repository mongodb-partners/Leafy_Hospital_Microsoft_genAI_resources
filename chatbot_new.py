import os
import base64
import uuid
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI as AzureOpenAIClient
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI, AzureOpenAI, AzureChatOpenAI
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
import asyncio
from datetime import datetime

import pymongo

#hardcoded to run as standalone
HARD_CODED_DIAGNOSTIC_REPORT_PATIENT = "Patient/1330"

load_dotenv()
MONGODB_CONNECTION_STRING = os.environ.get('MONGODB_CONNECTION_STRING')
client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)

# Provide the database and collection
database = client['your_database']
collection = database['your_collection']


# Function to connect to MongoDB and return a collection
def connect_mongodb(col):
    return database[col]
#replace with your embedding model name
def get_embedding(text, model="text-embedding-ada-002"):
    clientAzureOpenAI = AzureOpenAIClient(
        api_key=os.environ["AZURE_OPENAI_API_KEY"], 
        api_version=os.environ["OPENAI_API_VERSION"], 
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )
    text = text.replace("\n", " ")
    response = clientAzureOpenAI.embeddings.create(input=[text], model=model)
    embedding = response['choices'][0]['embedding'] if isinstance(response, dict) else response.data[0].embedding
    return embedding

# Function to find similar documents based on embeddings
def find_similar_documents(embedding, k):
    print("Searching for similar documents in <RAG_collection>...")
    collection = connect_mongodb(col="RAG_collection")
    documents = list(collection.aggregate([{
        "$vectorSearch": {
            "index": "default",
            "path": "vector_embedding",
            "queryVector": embedding, 
            "numCandidates": 200, 
            "limit": k
        }
    }]))

    print(f"Found {len(documents)} similar documents in RAG_collection.")
    
    return documents

# Get teh basic patient details from the patients collection
def load_patient_info(patient_name, patient_last_name, patient_birth_date):
#replace with your patients collection name    
    collection = connect_mongodb(col="patients")
    patient_doc = collection.find_one({"name.0.given.0": patient_name, 
                                       "name.0.family": patient_last_name, 
                                       "birthDate": datetime.strptime(patient_birth_date, "%Y-%m-%d")})
    return str(patient_doc)
# Get the patient history from the Reports collection
def load_patient_history(patient):
    #replace with your reports collection name  
    collection = connect_mongodb(col="Reports")
    pipeline = [
        {
            "$match": {
                "metadata.searchParameters": {
                    "$elemMatch": {"key": "patient", "value": patient}
                }
            }
        },
        {
            "$project": {
                "resource.presentedForm.data": 1,
                "resource.issued": 1,
                "_id": 0
            }
        },
        {
            "$sort": {
                "resource.issued": -1
            }
        },
        {
            "$limit": 10
        }
    ]
    patient_docs = collection.aggregate(pipeline)
    if not patient_docs:
        return ""
    full_patient_history = ""
    for patient_doc in patient_docs:
        try:
            patient_data = patient_doc["resource"]["presentedForm"][0]["data"]
        except KeyError:
            # discard doc without data
            continue
        decoded_data = base64.b64decode(patient_data).decode("utf-8")
        full_patient_history += decoded_data
        full_patient_history += "\n"
    
    # print("full_patient_history before summary BEGIN:")
    # print(full_patient_history)
    # print("full_patient_history before summary END")
    # get summary from LLM
    summary_prompt = "Below is the medical history record of a patient based on their latest 10 encounters with the doctor. Please summarize this record in less than 300 words, and try to keep all key information like patient's background, medical condition, dignosis, treatment history etc., but do not mention the name or age:"
    summary_prompt += "\n"
    summary_prompt += full_patient_history
    summary = chat(summary_prompt)
    return summary

def chat(prompt) -> str:
    kernel = sk.Kernel()
    api_key = os.environ["OPENAI_API_KEY"]
    service_id = "chatbot"
    # kernel.add_service(
    #     OpenAIChatCompletion(service_id=service_id, ai_model_id="gpt-3.5-turbo-1106", api_key=api_key),
    # )
    kernel.add_service(AzureChatCompletion(
        service_id=service_id,
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["OPENAI_API_VERSION"], 
    ))
    execution_settings = kernel.get_service(service_id).instantiate_prompt_execution_settings(service_id=service_id)
    
    prompt_function = kernel.add_function(
        function_name="chatbot_function", plugin_name="chatbot_plugin", prompt=prompt, prompt_execution_settings=execution_settings
    )
    
    response = str(asyncio.run(kernel.invoke(prompt_function)))
    
    return response

def answer_question_with_history(user_question, chat_history, patient_name, patient_last_name, patient_birth_date):
    print(f"Received question: {user_question}")

    aoai_llm = AzureChatOpenAI(
                  azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                  deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                  openai_api_version=os.environ["OPENAI_API_VERSION"], 
                  openai_api_key=os.environ["AZURE_OPENAI_API_KEY"]
                  )
    
    context = ""
    context_sources = []

    question_embedding = get_embedding(text=user_question)   
    documents = find_similar_documents(question_embedding, 3)
    df = pd.DataFrame(documents)
    for index, row in df[0:50].iterrows():
        context = context + " " + row.text_chunk
        context_sources.append(f"{row.get('source', {}).get('filename', 'Unknown Source')} - Page {row.get('source', {}).get('page_number', 'Unknown Page')}\n")

    patient_info = load_patient_info(patient_name, patient_last_name, patient_birth_date)
    patient_history = load_patient_history(HARD_CODED_DIAGNOSTIC_REPORT_PATIENT)
    
    
    prompt_text = f"""
    You are a chatbot used by doctors to get information regarding their patients details, their condition and treatment given so far. Using the following sections to answer the question using the information most relevant to the question. 
    “Context section” contains information from pdfs on cancer in general and its treatment methods and to be used to answer any general cancer related question. 
    “Patient info” is the personal information of the patient and to be used to answer any questions asked about the patient herself like her name, age, etc. 
    “Patient history” is the details of the patient's background, disease condition and treatment prescribed to her over the years. Use this to answer any questions asked about her disease, condition and treatment.
    Donot make up any information, try to answer from the context sections, patient info and patient history only, BUT DONOT mention in your response which section was referred to provide the answer. 
    For any question not at all related to patients or cancer in general, politely decline from answering. If you dont have exact information but related one, you can mention that. But dont make up any information or give unrelated answer.
    For example, if the current condition of the patient is asked and you dont have that information, you can say "I dont have the current condition of the patient, but she was diagnosed with cancer in 2018 and later chemotherapy was started which showed improvement in 2020".
    Your response should be less than 150 words and to the point. Dont provide any unnecessary information.

    Context sections:
    {context}
    
    Patient Info:
    {patient_info}
    
    Patient History:
    {patient_history}
    
    Chat history:
    {chat_history}

    Question:
    {user_question}

    Answer:
    """
    
    print("=====FINAL PROMPT BEGIN=====")
    print(prompt_text)
    print("=====FINAL PROMPT END=====")
    
    response = chat(prompt_text)
    
    # response = aoai_llm.invoke(prompt_text).content
    
    print(f"Response: {response}")
    print(f"Context sources: {context_sources}")
    return response, context_sources

def load_chat_history(conversation_id):
    collection = connect_mongodb(col="chatHistory")
    retrieved_doc = collection.find_one({"id": conversation_id})
    if retrieved_doc:
        return retrieved_doc["chat_history"]
    else:
        return []

def save_chat_history(conversation_id, chat_history: list[dict]):
    collection = connect_mongodb(col="chatHistory")
    chat_history_data = {"id": conversation_id, "chat_history": chat_history}
    collection.find_one_and_update({"id": conversation_id}, {"$set": chat_history_data}, upsert=True)

# main entry of the chatbot
def invoke_chatbot(prompt, patient_name, patient_last_name, patient_birth_date, conversation_id=None):
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        chat_history = []
    else:
        chat_history = load_chat_history(conversation_id)
    response, context_sources = answer_question_with_history(prompt, chat_history, patient_name, patient_last_name, patient_birth_date)
    chat_history.append({"role": "user", "content": prompt})
    chat_history.append({"role": "assistant", "content": response})
    save_chat_history(conversation_id, chat_history)
    return response, conversation_id, context_sources

if __name__ == "__main__":
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit", "bye"]:
            break
        response = invoke_chatbot(prompt,"Flora", "Harper", "1966-01-18")
        print(f"Chatbot: {response}")

