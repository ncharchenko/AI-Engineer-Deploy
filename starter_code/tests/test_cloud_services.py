import unittest
from types import SimpleNamespace

from cloud_services import (
    PRODUCT_DATABASE_UNAVAILABLE_MESSAGE,
    ProductDatabaseUnavailableError,
    create_redis_chat_history,
    initialize_langfuse_client,
    initialize_qdrant_store,
    search_qdrant_store,
)


class FakeQdrantClient:
    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        collection_exists: bool,
        point_count: int,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self._collection_exists = collection_exists
        self.point_count = point_count
        self.created_collection = None
        self.deleted_collection = None

    def collection_exists(self, *, collection_name: str) -> bool:
        self.checked_collection = collection_name
        return self._collection_exists

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        self.created_collection = (collection_name, vectors_config)
        self._collection_exists = True

    def count(self, *, collection_name: str, exact: bool) -> SimpleNamespace:
        self.counted_collection = (collection_name, exact)
        return SimpleNamespace(count=self.point_count)

    def delete_collection(self, *, collection_name: str) -> None:
        self.deleted_collection = collection_name
        self._collection_exists = False
        self.point_count = 0


class FakeVectorStore:
    existing_collection_arguments = None

    def __init__(self, *, client: object, collection_name: str, embedding: object) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding
        self.added_documents = None

    @classmethod
    def from_existing_collection(cls, **kwargs: object) -> "FakeVectorStore":
        cls.existing_collection_arguments = kwargs
        return cls(
            client="existing-client",
            collection_name=str(kwargs["collection_name"]),
            embedding=kwargs["embedding"],
        )

    def add_documents(self, *, documents: list[object]) -> None:
        self.added_documents = documents
        if isinstance(self.client, FakeQdrantClient):
            self.client.point_count = len(documents)


class FailingPopulationStore(FakeVectorStore):
    def add_documents(self, *, documents: list[object]) -> None:
        raise RuntimeError(
            "provider failed with api_key=SENSITIVE_TOKEN"
        )


class CapturingClient:
    def __init__(self, **kwargs: object) -> None:
        self.arguments = kwargs


class FailingQdrantClient:
    def __init__(self, *, url: str, api_key: str) -> None:
        pass

    def collection_exists(self, *, collection_name: str) -> bool:
        raise ConnectionError("Qdrant is unreachable")


class FailingVectorStore:
    def similarity_search(self, query: str, *, k: int) -> list[object]:
        raise TimeoutError("Qdrant search timed out")


class QdrantCloudTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeVectorStore.existing_collection_arguments = None
        self.client = None
        self.point_count = 1

    def client_factory(self, *, url: str, api_key: str) -> FakeQdrantClient:
        self.client = FakeQdrantClient(
            url=url,
            api_key=api_key,
            collection_exists=self.collection_exists,
            point_count=self.point_count,
        )
        return self.client

    def test_reuses_existing_cloud_collection_with_credentials(self) -> None:
        self.collection_exists = True
        embedding = object()

        initialize_qdrant_store(
            documents=[object()],
            embeddings_model=embedding,
            qdrant_url="https://qdrant.example.com",
            qdrant_api_key="test-key",
            client_factory=self.client_factory,
            vector_store_class=FakeVectorStore,
        )

        self.assertEqual(
            FakeVectorStore.existing_collection_arguments,
            {
                "url": "https://qdrant.example.com",
                "api_key": "test-key",
                "embedding": embedding,
                "collection_name": "smartphones",
            },
        )

    def test_creates_and_populates_missing_cloud_collection(self) -> None:
        self.collection_exists = False
        self.point_count = 0
        documents = [object()]

        store = initialize_qdrant_store(
            documents=documents,
            embeddings_model=object(),
            qdrant_url="https://qdrant.example.com",
            qdrant_api_key="test-key",
            client_factory=self.client_factory,
            vector_store_class=FakeVectorStore,
        )

        self.assertEqual(self.client.url, "https://qdrant.example.com")
        self.assertEqual(self.client.api_key, "test-key")
        self.assertEqual(self.client.created_collection[0], "smartphones")
        self.assertEqual(store.added_documents, documents)
        self.assertEqual(
            self.client.counted_collection,
            ("smartphones", True),
        )

    def test_rejects_incomplete_existing_collection(self) -> None:
        self.collection_exists = True
        self.point_count = 0

        with self.assertRaises(ProductDatabaseUnavailableError) as raised:
            initialize_qdrant_store(
                documents=[object()],
                embeddings_model=object(),
                qdrant_url="https://qdrant.example.com",
                qdrant_api_key="test-key",
                client_factory=self.client_factory,
                vector_store_class=FakeVectorStore,
            )

        self.assertEqual(raised.exception.operation, "verify_collection")
        self.assertEqual(
            raised.exception.error_type,
            "_IncompleteCollectionError",
        )
        self.assertIsNone(FakeVectorStore.existing_collection_arguments)

    def test_removes_new_collection_when_population_fails(self) -> None:
        self.collection_exists = False
        self.point_count = 0

        with self.assertRaises(ProductDatabaseUnavailableError) as raised:
            initialize_qdrant_store(
                documents=[object()],
                embeddings_model=object(),
                qdrant_url="https://qdrant.example.com",
                qdrant_api_key="test-key",
                client_factory=self.client_factory,
                vector_store_class=FailingPopulationStore,
            )

        self.assertEqual(raised.exception.operation, "populate_collection")
        self.assertEqual(raised.exception.error_type, "RuntimeError")
        self.assertEqual(self.client.deleted_collection, "smartphones")
        self.assertFalse(self.client._collection_exists)
        self.assertIsNone(raised.exception.__cause__)

    def test_reports_unavailable_when_qdrant_initialization_fails(self) -> None:
        with self.assertRaisesRegex(
            ProductDatabaseUnavailableError,
            PRODUCT_DATABASE_UNAVAILABLE_MESSAGE,
        ):
            initialize_qdrant_store(
                documents=[object()],
                embeddings_model=object(),
                qdrant_url="https://qdrant.example.com",
                qdrant_api_key="test-key",
                client_factory=FailingQdrantClient,
                vector_store_class=FakeVectorStore,
            )

    def test_reports_unavailable_when_qdrant_search_fails(self) -> None:
        with self.assertRaisesRegex(
            ProductDatabaseUnavailableError,
            PRODUCT_DATABASE_UNAVAILABLE_MESSAGE,
        ) as raised:
            search_qdrant_store(
                vector_store=FailingVectorStore(),
                query="iPhone",
            )

        self.assertEqual(raised.exception.operation, "similarity_search")
        self.assertEqual(raised.exception.error_type, "TimeoutError")
        self.assertIsNone(raised.exception.__cause__)


class CloudClientWiringTests(unittest.TestCase):
    def test_passes_cloud_url_to_redis_history(self) -> None:
        history = create_redis_chat_history(
            session_id="session-123",
            redis_url="rediss://default:password@redis.example.com:12345/0",
            history_class=CapturingClient,
        )

        self.assertEqual(
            history.arguments,
            {
                "session_id": "session-123",
                "redis_url": "rediss://default:password@redis.example.com:12345/0",
                "ttl": 3600,
            },
        )

    def test_initializes_langfuse_with_cloud_project(self) -> None:
        client = initialize_langfuse_client(
            public_key="test-public-key",
            secret_key="test-secret-key",
            host="https://cloud.langfuse.com",
            client_class=CapturingClient,
        )

        self.assertEqual(
            client.arguments,
            {
                "public_key": "test-public-key",
                "secret_key": "test-secret-key",
                "base_url": "https://cloud.langfuse.com",
            },
        )


if __name__ == "__main__":
    unittest.main()
