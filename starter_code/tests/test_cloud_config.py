import unittest

from pydantic import ValidationError

from cloud_config import load_app_config


VALID_ENVIRONMENT = {
    "OPENAI_MODEL": "test-chat-model",
    "OPENAI_EMBEDDINGS_MODEL": "test-embeddings-model",
    "LITELLM_API_KEY": "test-litellm-key",
    "OPENAI_BASE_URL": "https://litellm.example.com/openai",
    "QDRANT_URL": "https://qdrant.example.com",
    "QDRANT_API_KEY": "test-qdrant-key",
    "REDIS_URL": "redis://default:password@redis.example.com:12345/0",
    "LANGFUSE_HOST": "https://cloud.langfuse.com",
    "LANGFUSE_PUBLIC_KEY": "test-langfuse-public-key",
    "LANGFUSE_SECRET_KEY": "test-langfuse-secret-key",
}


class CloudConfigTests(unittest.TestCase):
    def test_loads_all_cloud_service_settings(self) -> None:
        config = load_app_config(VALID_ENVIRONMENT)

        self.assertEqual(config.qdrant_url, VALID_ENVIRONMENT["QDRANT_URL"])
        self.assertEqual(config.redis_url, VALID_ENVIRONMENT["REDIS_URL"])
        self.assertEqual(config.langfuse_host, VALID_ENVIRONMENT["LANGFUSE_HOST"])

    def test_missing_cloud_setting_fails_validation(self) -> None:
        environment = VALID_ENVIRONMENT.copy()
        del environment["QDRANT_API_KEY"]

        with self.assertRaises(ValidationError):
            load_app_config(environment)

    def test_tls_redis_url_is_supported(self) -> None:
        environment = VALID_ENVIRONMENT | {
            "REDIS_URL": "rediss://default:password@redis.example.com:12345/0"
        }

        config = load_app_config(environment)

        self.assertEqual(config.redis_url, environment["REDIS_URL"])

    def test_non_redis_url_is_rejected(self) -> None:
        environment = VALID_ENVIRONMENT | {
            "REDIS_URL": "https://redis.example.com:12345/0"
        }

        with self.assertRaises(ValidationError):
            load_app_config(environment)

    def test_missing_langfuse_url_fails_validation(self) -> None:
        environment = VALID_ENVIRONMENT.copy()
        del environment["LANGFUSE_HOST"]

        with self.assertRaises(ValidationError):
            load_app_config(environment)

    def test_langfuse_base_url_does_not_replace_required_host(self) -> None:
        environment = VALID_ENVIRONMENT.copy()
        del environment["LANGFUSE_HOST"]
        environment["LANGFUSE_BASE_URL"] = "https://langfuse.example.com"

        with self.assertRaises(ValidationError):
            load_app_config(environment)

    def test_missing_openai_base_url_fails_validation(self) -> None:
        environment = VALID_ENVIRONMENT.copy()
        del environment["OPENAI_BASE_URL"]

        with self.assertRaises(ValidationError):
            load_app_config(environment)

    def test_strips_surrounding_whitespace_from_required_values(self) -> None:
        environment = VALID_ENVIRONMENT | {
            "QDRANT_URL": "  https://qdrant.example.com  ",
            "QDRANT_API_KEY": "  test-qdrant-key  ",
        }

        config = load_app_config(environment)

        self.assertEqual(config.qdrant_url, "https://qdrant.example.com")
        self.assertEqual(config.qdrant_api_key, "test-qdrant-key")

    def test_rejects_whitespace_only_required_value(self) -> None:
        environment = VALID_ENVIRONMENT | {"QDRANT_API_KEY": "   "}

        with self.assertRaises(ValidationError):
            load_app_config(environment)

    def test_rejects_control_character_in_credential(self) -> None:
        environment = VALID_ENVIRONMENT | {
            "QDRANT_API_KEY": "test-key\ninjected-value"
        }

        with self.assertRaises(ValidationError):
            load_app_config(environment)

    def test_rejects_control_character_in_url(self) -> None:
        environment = VALID_ENVIRONMENT | {
            "QDRANT_URL": "https://qdrant.example.com\tunexpected"
        }

        with self.assertRaises(ValidationError):
            load_app_config(environment)


if __name__ == "__main__":
    unittest.main()
