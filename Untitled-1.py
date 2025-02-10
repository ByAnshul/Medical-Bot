# %%
print("ds")


# %%
%pwd

# %%
import os 

# %%
os.chdir("../")

# %%
%pwd

# %%
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# %%
#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

# %%
extracted_data=load_pdf_file(data='Data/')


# %%
# print(extracted_data)

# %%
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 ,chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# %%
text_chunks = text_split(extracted_data)
print("length of text chunks :",len(text_chunks))

# %%


# %%
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# %%
embeddings = download_hugging_face_embeddings()



# %%
print("ok") 

# %%
embeddings = download_hugging_face_embeddings()


# %%
query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))


# %%
from dotenv import load_dotenv
import os

load_dotenv("D:\Major Project\YT bot 2\Medical-Bot\.env")  # Example: "D:/Major Project/YT bot 2/.env"


# %%


# %%
import os
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY="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"
print("OPENAI_API_KEY:", OPENAI_API_KEY)



# %%
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# %%



pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# %%
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# %%
# Embed each chunk and upsert the embeddings into your Pinecone index.
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

# %%
#Load Existing index
from langchain_pinecone import PineconeVectorStore
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# %%
docsearch


# %%
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# %%
retrieved_docs = retriever.invoke("What is Tuberculosis?")


# %%
retrieved_docs

# %%
from langchain_openai import OpenAI
llm = OpenAI(temperature=0.4, max_tokens=500)

# %%
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# %%
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print(OPENAI_API_KEY)

# %%
# Require Paid OPENAI API
response = rag_chain.invoke({"input": "what is Acromegaly and gigantism?"})
print(response["answer"])

# %%
# # It require 9 gb file locally to download 
# from transformers import pipeline

# hf_api_key = "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"

# generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1", token=hf_api_key)
# print(generator("What is Acromegaly?", max_length=100))

# %%
# import requests

# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
# headers = {"Authorization": f"Bearer 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"}

# data = {"inputs": "Explain diabetes in simple terms."}
# response = requests.post(API_URL, headers=headers, json=data)
# print(response.json())


# %% [markdown]
#     ------------------------------------------This is the Review 1-------------------------------------------------------------

# %%
import requests

api_key = "c1b1fc1f38efb1f7cf5b4d82d32aa20948bc40b273b7b0784b90ca4a3dee0ffd"  # Get from https://together.ai/
url = "https://api.together.xyz/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

data = {
    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "messages": [{"role": "user", "content": "What is asthma "}]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())  # Debugging step

print(response.json()["choices"][0]["message"]["content"])


# %%
import requests
from langchain.chat_models import ChatOpenAI  # Used for TogetherAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# API Key for Together AI
TOGETHER_API_KEY = "c1b1fc1f38efb1f7cf5b4d82d32aa20948bc40b273b7b0784b90ca4a3dee0ffd"

# Set Up Together AI LLM using ChatOpenAI
llm = ChatOpenAI(
    openai_api_key=TOGETHER_API_KEY,  
    openai_api_base="https://api.together.xyz/v1",  # Base URL for Together AI
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.4,
    max_tokens=500
)

# Define Prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Define Retrieval Chain (assuming retriever is already defined)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Invoke RAG Chain
response = rag_chain.invoke({"input": "What is CORONA "})
print(response["answer"])


# %%
import requests
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Together API Setup
API_KEY = "c1b1fc1f38efb1f7cf5b4d82d32aa20948bc40b273b7b0784b90ca4a3dee0ffd"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Load Existing Pinecone Index
index_name = "medicalbot"
retriever = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings).as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)
def query_medical_chatbot(user_query):
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(user_query)
    
    # Extract text from documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Format the request
    system_prompt = (
        "You are an expert medical assistant. Use the retrieved context below "
        "to answer the user's question accurately. If the answer is unknown, say 'I don't know'."
        "\n\nRetrieved Context:\n"
        f"{context}"
    )

    # API request to Together
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    }

    response = requests.post(TOGETHER_URL, headers=HEADERS, json=payload)
    
    # Handle API response
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"


# Example query
response = query_medical_chatbot("use of mouth")
print(response)


# %%



