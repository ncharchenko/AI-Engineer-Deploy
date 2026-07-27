from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import Langfuse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


PRODUCT_DATABASE_UNAVAILABLE_MESSAGE = (
    "The smartphone product database is currently unavailable."
)


class ProductDatabaseUnavailableError(RuntimeError):
    """Raised when the smartphone vector database cannot serve a request."""

    def __init__(
        self,
        *,
        operation: str,
        error_type: str,
        collection_name: str = "smartphones",
        cleanup_error_type: str | None = None,
    ) -> None:
        super().__init__(PRODUCT_DATABASE_UNAVAILABLE_MESSAGE)
        self.operation = operation
        self.error_type = error_type
        self.collection_name = collection_name
        self.cleanup_error_type = cleanup_error_type


class _IncompleteCollectionError(RuntimeError):
    """Internal marker for a collection that is not ready to serve searches."""


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
    if not documents:
        raise ProductDatabaseUnavailableError(
            operation="validate_dataset",
            error_type="EmptyDatasetError",
            collection_name=collection_name,
        )

    client = None
    created_collection = False
    collection_verified = False
    operation = "initialize_client"

    try:
        client = client_factory(url=qdrant_url, api_key=qdrant_api_key)

        operation = "check_collection"
        if client.collection_exists(collection_name=collection_name):
            operation = "verify_collection"
            count_result = client.count(
                collection_name=collection_name,
                exact=True,
            )
            if count_result.count < len(documents):
                raise _IncompleteCollectionError(
                    "Existing collection is not fully populated"
                )

            operation = "open_collection"
            return vector_store_class.from_existing_collection(
                url=qdrant_url,
                api_key=qdrant_api_key,
                embedding=embeddings_model,
                collection_name=collection_name,
            )

        operation = "create_collection"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )
        created_collection = True

        operation = "populate_collection"
        vector_store = vector_store_class(
            client=client,
            collection_name=collection_name,
            embedding=embeddings_model,
        )
        vector_store.add_documents(documents=documents)

        operation = "verify_population"
        count_result = client.count(
            collection_name=collection_name,
            exact=True,
        )
        if count_result.count < len(documents):
            raise _IncompleteCollectionError(
                "New collection is not fully populated"
            )

        collection_verified = True
        return vector_store
    except Exception as exc:
        cleanup_error_type = None
        if created_collection and not collection_verified and client is not None:
            try:
                client.delete_collection(collection_name=collection_name)
            except Exception as cleanup_exc:
                cleanup_error_type = type(cleanup_exc).__name__

        raise ProductDatabaseUnavailableError(
            operation=operation,
            error_type=type(exc).__name__,
            collection_name=collection_name,
            cleanup_error_type=cleanup_error_type,
        ) from None


def search_qdrant_store(
    *,
    vector_store: QdrantVectorStore,
    query: str,
    k: int = 1,
) -> list[Document]:
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as exc:
        raise ProductDatabaseUnavailableError(
            operation="similarity_search",
            error_type=type(exc).__name__,
        ) from None
