from __future__ import annotations

from fastapi.testclient import TestClient

from scholarpath import api
from tests.conftest import example


def test_recommend_endpoint() -> None:
    api.get_graph.cache_clear()
    client = TestClient(api.app)
    response = client.post("/recommend", json={"text": example("student_3.txt")})
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["field_of_study"] == "Business Administration"
    assert body["opportunities"]
    assert body["report_markdown"].startswith("# ScholarPath Report")


def test_recommend_rejects_empty_text() -> None:
    client = TestClient(api.app)
    assert client.post("/recommend", json={"text": "hi"}).status_code == 422
