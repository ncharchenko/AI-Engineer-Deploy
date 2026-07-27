import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import cloud_services


TEST_ENVIRONMENT = {
    "OPENAI_MODEL": "test-chat-model",
    "OPENAI_EMBEDDINGS_MODEL": "test-embeddings-model",
    "LITELLM_API_KEY": "test-litellm-key",
    "OPENAI_BASE_URL": "https://litellm.invalid/openai",
    "QDRANT_URL": "https://qdrant.invalid",
    "QDRANT_API_KEY": "test-qdrant-key",
    "REDIS_URL": "redis://localhost:6379/0",
    "LANGFUSE_HOST": "https://langfuse.invalid",
    "LANGFUSE_PUBLIC_KEY": "test-public-key",
    "LANGFUSE_SECRET_KEY": "test-secret-key",
    "LANGFUSE_TRACING_ENABLED": "false",
}
os.environ.update(TEST_ENVIRONMENT)
os.environ.pop("APP_SECRET_ID", None)


class FakePrompt:
    def get_langchain_prompt(self) -> list[tuple[str, str]]:
        return [("system", "test")]


class FakeSpan:
    def update(self, **kwargs: object) -> None:
        pass


class FakeLangfuse:
    def get_prompt(self, name: str) -> FakePrompt:
        return FakePrompt()

    @contextmanager
    def start_as_current_observation(self, **kwargs: object):
        yield FakeSpan()


original_langfuse_initializer = cloud_services.initialize_langfuse_client
cloud_services.initialize_langfuse_client = lambda **kwargs: FakeLangfuse()
try:
    import main
finally:
    cloud_services.initialize_langfuse_client = original_langfuse_initializer


@contextmanager
def no_op_attributes(**kwargs: object):
    yield


class AllowedRails:
    def generate(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(output_data={"allowed": True}, response=[])


class FakeHistory:
    def __init__(self) -> None:
        self.messages: list[object] = []
        self.saved: list[object] = []

    def add_message(self, message: object) -> None:
        self.saved.append(message)


class SecretBearingFailureStore:
    def similarity_search(self, query: str, *, k: int) -> list[Document]:
        raise ConnectionError(
            "https://user:SENSITIVE_PASSWORD@qdrant.invalid "
            "api_key=SENSITIVE_TOKEN"
        )


def smartphone_tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "SmartphoneInfo",
                "args": {"model": "Phone A"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


class AskEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(main.app, raise_server_exceptions=False)

    def setUp(self) -> None:
        self.history = FakeHistory()
        self.context_chain = Mock()
        self.context_chain.invoke.return_value = smartphone_tool_call()
        self.review_chain = Mock()
        self.review_chain.invoke.return_value = AIMessage(
            content="product response"
        )

        patchers = [
            patch.object(main, "propagate_attributes", no_op_attributes),
            patch.object(
                main,
                "input_rails",
                SimpleNamespace(rails=AllowedRails()),
            ),
            patch.object(
                main,
                "create_redis_chat_history",
                return_value=self.history,
            ),
            patch.object(main, "context_chain", self.context_chain),
            patch.object(main, "review_chain", self.review_chain),
            patch.object(main, "embed_documents"),
            patch.object(main, "search_qdrant_store"),
        ]
        started_patchers = [patcher.start() for patcher in patchers]
        for patcher in patchers:
            self.addCleanup(patcher.stop)

        self.redis_factory = started_patchers[2]
        self.embed_documents = started_patchers[5]
        self.search_qdrant_store = started_patchers[6]

        self.payload = {
            "user_input": "Compare Phone A",
            "user_id": "user-1",
            "session_id": "session-1",
        }

    def test_successful_tool_retrieval_returns_200_and_persists(self) -> None:
        self.embed_documents.return_value = object()
        self.search_qdrant_store.return_value = [
            Document(page_content="Model: Phone A")
        ]

        response = self.client.post("/ask", json=self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "product response"})
        self.review_chain.invoke.assert_called_once()
        self.assertEqual(len(self.history.saved), 2)
        conversation = self.review_chain.invoke.call_args.args[0]["conversation"]
        self.assertTrue(
            any(
                getattr(message, "content", None) == "Model: Phone A"
                for message in conversation
            )
        )

    def test_legitimate_no_result_returns_normal_200(self) -> None:
        self.embed_documents.return_value = object()
        self.search_qdrant_store.return_value = []
        self.review_chain.invoke.return_value = AIMessage(
            content="no matching product"
        )

        response = self.client.post("/ask", json=self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"response": "no matching product"},
        )
        self.review_chain.invoke.assert_called_once()
        self.assertEqual(len(self.history.saved), 2)
        conversation = self.review_chain.invoke.call_args.args[0]["conversation"]
        self.assertTrue(
            any(
                getattr(message, "content", None)
                == "Could not find information for the specified model."
                for message in conversation
            )
        )

    def test_typed_tool_failure_returns_503_without_review_or_redis(self) -> None:
        self.embed_documents.side_effect = (
            cloud_services.ProductDatabaseUnavailableError(
                operation="check_collection",
                error_type="ConnectionError",
            )
        )

        response = self.client.post("/ask", json=self.payload)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "detail": (
                    cloud_services.PRODUCT_DATABASE_UNAVAILABLE_MESSAGE
                )
            },
        )
        self.review_chain.invoke.assert_not_called()
        self.assertEqual(self.history.saved, [])

    def test_unexpected_tool_failure_returns_500_without_review_or_redis(
        self,
    ) -> None:
        self.embed_documents.side_effect = RuntimeError(
            "unexpected tool failure"
        )

        response = self.client.post("/ask", json=self.payload)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Failed to generate a response."},
        )
        self.review_chain.invoke.assert_not_called()
        self.assertEqual(self.history.saved, [])

    def test_final_review_failure_returns_500_without_redis(self) -> None:
        self.embed_documents.return_value = object()
        self.search_qdrant_store.return_value = [
            Document(page_content="Model: Phone A")
        ]
        self.review_chain.invoke.side_effect = RuntimeError(
            "review provider unavailable"
        )

        response = self.client.post("/ask", json=self.payload)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {"detail": "Failed to generate a response."},
        )
        self.review_chain.invoke.assert_called_once()
        self.assertEqual(self.history.saved, [])

    def test_provider_secrets_are_not_written_to_logs(self) -> None:
        self.embed_documents.return_value = SecretBearingFailureStore()
        self.search_qdrant_store.side_effect = (
            cloud_services.search_qdrant_store
        )

        with self.assertLogs(main.logger, level="ERROR") as captured:
            response = self.client.post("/ask", json=self.payload)

        logs = "\n".join(captured.output)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("SENSITIVE_TOKEN", logs)
        self.assertNotIn("SENSITIVE_PASSWORD", logs)
        self.assertNotIn("qdrant.invalid", logs)
        self.assertIn("operation=similarity_search", logs)
        self.assertIn("exception_type=ConnectionError", logs)
        self.review_chain.invoke.assert_not_called()
        self.assertEqual(self.history.saved, [])

    def test_failed_population_cannot_become_false_not_found(self) -> None:
        state = SimpleNamespace(
            exists=False,
            point_count=0,
            population_attempts=0,
        )

        class StatefulClient:
            def __init__(self, **kwargs: object) -> None:
                pass

            def collection_exists(self, **kwargs: object) -> bool:
                return state.exists

            def create_collection(self, **kwargs: object) -> None:
                state.exists = True

            def count(self, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(count=state.point_count)

            def delete_collection(self, **kwargs: object) -> None:
                state.exists = False
                state.point_count = 0

        class FailsOnceStore:
            def __init__(
                self,
                *,
                client: StatefulClient,
                collection_name: str,
                embedding: object,
            ) -> None:
                self.client = client

            @classmethod
            def from_existing_collection(
                cls,
                **kwargs: object,
            ) -> "FailsOnceStore":
                raise AssertionError("incomplete collection was reused")

            def add_documents(
                self,
                *,
                documents: list[Document],
            ) -> None:
                state.population_attempts += 1
                if state.population_attempts == 1:
                    raise RuntimeError("embedding provider unavailable")
                state.point_count = len(documents)

            def similarity_search(
                self,
                query: str,
                *,
                k: int,
            ) -> list[Document]:
                return [Document(page_content="Model: Phone A")]

        documents = [Document(page_content="Model: Phone A")]

        def initialize_store(json_path: str) -> object:
            return cloud_services.initialize_qdrant_store(
                documents=documents,
                embeddings_model=object(),
                qdrant_url="https://qdrant.invalid",
                qdrant_api_key="test-key",
                client_factory=StatefulClient,
                vector_store_class=FailsOnceStore,
            )

        first_history = FakeHistory()
        second_history = FakeHistory()
        self.redis_factory.side_effect = [first_history, second_history]
        self.embed_documents.side_effect = initialize_store
        self.search_qdrant_store.side_effect = (
            cloud_services.search_qdrant_store
        )

        first_response = self.client.post("/ask", json=self.payload)
        second_response = self.client.post("/ask", json=self.payload)

        self.assertEqual(first_response.status_code, 503)
        self.assertEqual(first_history.saved, [])
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(
            second_response.json(),
            {"response": "product response"},
        )
        self.review_chain.invoke.assert_called_once()
        self.assertEqual(len(second_history.saved), 2)
        self.assertEqual(state.population_attempts, 2)
        self.assertTrue(state.exists)
        self.assertEqual(state.point_count, 1)
        conversation = self.review_chain.invoke.call_args.args[0]["conversation"]
        self.assertTrue(
            any(
                getattr(message, "content", None) == "Model: Phone A"
                for message in conversation
            )
        )


if __name__ == "__main__":
    unittest.main()
