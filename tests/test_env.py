import pytest
from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "tasks" in response.json()
    assert len(response.json()["tasks"]) == 3

def test_tasks():
    response = client.get("/tasks")
    assert response.status_code == 200
    assert "tasks" in response.json()
    assert len(response.json()["tasks"]) == 3
    for task in response.json()["tasks"]:
        assert "id" in task
        assert "title" in task
        assert "description" in task

def test_reset():
    response = client.post("/reset", json={"task_id": "easy"})
    assert response.status_code == 200
    assert "observation" in response.json()
    assert "reward" in response.json()
    assert "done" in response.json()
