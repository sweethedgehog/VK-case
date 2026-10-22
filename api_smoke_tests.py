"""HTTP smoke tests that do not require Gemini credentials or model downloads."""
from fastapi.testclient import TestClient
import server


def run() -> None:
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        home = client.get("/")
        assert home.status_code == 200 and "Навигатор ИВТ" in home.text
        assert client.get("/api/status").status_code == 200
        courses = client.get("/api/courses")
        assert courses.status_code == 200 and "courses" in courses.json()
        # Invalid input is rejected before model startup and never spends API quota.
        answer = client.post("/api/ask", json={"question": "", "session_id": "smoke-test"})
        assert answer.status_code == 422
    print("HTTP smoke tests passed.")


if __name__ == "__main__":
    run()
