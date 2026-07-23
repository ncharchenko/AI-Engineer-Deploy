import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from health import router as health_router


class HealthEndpointTests(unittest.TestCase):
    def test_health_returns_stable_liveness_response(self) -> None:
        app = FastAPI()
        app.include_router(health_router)

        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
