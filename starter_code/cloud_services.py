from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import Langfuse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def initialize_langfuse_client(
    *,
    public_key: str,
    secret_key: str,
    host: str,
    client_class: type[Langfuse] = Langfuse,
) -> Langfuse:
    return client_class(
        public_key=public_key,
        secret_key=secret_key,
        base_url=host,
    )


def create_redis_chat_history(
    *,
    session_id: str,
    redis_url: str,
    ttl: int = 3600,
    history_class: type[RedisChatMessageHistory] = RedisChatMessageHistory,
) -> RedisChatMessageHistory:
    return history_class(
        session_id=session_id,
        redis_url=redis_url,
        ttl=ttl,
    )


def initialize_qdrant_store(
    *,
    documents: list[Document],
    embeddings_model: Embeddings,
    qdrant_url: str,
    qdrant_api_key: str,
    collection_name: str = "smartphones",
    client_factory: Callable[..., Any] = QdrantClient,
    vector_store_class: type[QdrantVectorStore] = QdrantVectorStore,
) -> QdrantVectorStore:
    client = client_factory(url=qdrant_url, api_key=qdrant_api_key)

    if client.collection_exists(collection_name=collection_name):
        return vector_store_class.from_existing_collection(
            url=qdrant_url,
            api_key=qdrant_api_key,
            embedding=embeddings_model,
            collection_name=collection_name,
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
    )
    vector_store = vector_store_class(
        client=client,
        collection_name=collection_name,
        embedding=embeddings_model,
    )
    vector_store.add_documents(documents=documents)

    return vector_store
