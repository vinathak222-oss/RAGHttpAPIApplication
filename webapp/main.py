import os
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

# Connect to the local Llamafile instead of Azure OpenAI
client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Embedding model, replaces langchain's OpenAIEmbeddings
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory Qdrant instance, replaces Azure Cognitive Search
qdrant = QdrantClient(":memory:")

COLLECTION_NAME = "top_wines"


class Body(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():
    """
    Runs once when the app starts. Loads the CSV, creates embeddings,
    and uploads them into the Qdrant collection so /ask has data to search.
    """
    load_data_into_qdrant()


def load_data_into_qdrant():
    df = pd.read_csv("wine-ratings.csv")

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )

    points = []
    for idx, row in df.iterrows():
        text = str(row.get("notes", "")) or str(row.to_dict())
        vector = encoder.encode(text).tolist()
        points.append(
            PointStruct(id=idx, vector=vector, payload=row.to_dict())
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Loaded {len(points)} rows into Qdrant collection '{COLLECTION_NAME}'")


@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the local Llamafile
    using Qdrant for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}


def search(query):
    """
    Send the query to Qdrant and return the top result
    """
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=encoder.encode(query).tolist(),
        limit=5,
    ).points

    result = hits[0].payload
    print(result)
    return result


def assistant(query, context):
    messages = [
        # Set the system characteristics for this chat bot
        {"role": "system", "content": "Assistant is a chatbot that helps you find the best wine for your taste."},

        # Set the query so that the chatbot can respond to it
        {"role": "user", "content": f"{query}\n\nHere is relevant context:\n{context}"},
    ]

    response = client.chat.completions.create(
        model="LLaMA_CPP",
        messages=messages,
    )
    return response.choices[0].message.content