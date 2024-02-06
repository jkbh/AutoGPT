from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import hashlib


class ChromaVectorStore:
    def __init__(self, store_path: str):
        self.client = PersistentClient(
            path=store_path, settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"]
        )

    def add(self, documents: list[str], metadatas: list[dict]) -> None:
        ids = [hashlib.sha256(doc.encode()).hexdigest()[:20] for doc in documents]
        collection = self.client.get_or_create_collection(
            "research_data_repository", embedding_function=self.embedding_function
        )

        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, documents: list[str], filters: dict = None) -> dict:
        collection = self.client.get_or_create_collection(
            "research_data_repository", embedding_function=self.embedding_function
        )
        return collection.query(query_texts=documents, n_results=5)
