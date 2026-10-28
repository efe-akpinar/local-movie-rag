import json
import logging

from movie_rag.generate import REFUSAL_TEXT, UNAVAILABLE_TEXT
from movie_rag.logs import JsonFormatter
from tests.test_pipeline import DownGenerator


def ask(client, question, **kwargs):
    return client.post("/ask", json={"question": question}, **kwargs)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "documents": 12, "provider": "fake"}


def test_ask_contract(client):
    response = ask(client, "Who directed Toy Story?")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"request_id", "answer", "citations", "refused", "retrieved", "timings_ms"}
    assert body["citations"] == ["Toy Story"] and body["refused"] is False
    assert len(body["retrieved"]) == 5
    top = body["retrieved"][0]
    assert (top["title"], top["year"]) == ("Toy Story", 1995) and isinstance(top["rerank_score"], float)
    assert set(body["timings_ms"]) == {"retrieve", "rerank", "generate", "total"}
    assert response.headers["x-request-id"] == body["request_id"]


def test_request_id_is_propagated(client):
    response = ask(client, "Who directed Jaws?", headers={"X-Request-ID": "abc123"})
    assert response.headers["x-request-id"] == "abc123" and response.json()["request_id"] == "abc123"


def test_post_cutoff_question_gets_the_exact_refusal(client):
    body = ask(client, "Who directed Oppenheimer (2023)?").json()
    assert body["answer"] == REFUSAL_TEXT
    assert body["refused"] is True and body["citations"] == []


def test_input_validation(client):
    assert ask(client, "").status_code == 422
    assert ask(client, "    ").status_code == 422
    assert ask(client, "x" * 301).status_code == 422
    assert ask(client, "x" * 300).status_code == 200
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": 42}).status_code == 422
    assert client.post("/ask", content="not json", headers={"content-type": "application/json"}).status_code == 422


def test_generation_outage_returns_503_with_a_friendly_message(make_client):
    response = ask(make_client(generator=DownGenerator()), "Who directed Jaws?")
    assert response.status_code == 503
    assert response.json()["answer"] == UNAVAILABLE_TEXT


def test_rate_limit_keys_on_the_address_cloud_run_appends(make_client):
    client = make_client(rate_limit_per_minute=3, trust_x_forwarded_for=True)
    # The caller controls everything before the last entry; rotating it must not reset the window.
    forged = [{"X-Forwarded-For": f"10.0.0.{i}, 198.51.100.7"} for i in range(4)]
    assert [ask(client, "Who directed Jaws?", headers=h).status_code for h in forged[:3]] == [200, 200, 200]
    limited = ask(client, "Who directed Jaws?", headers=forged[3])
    assert limited.status_code == 429
    assert 1 <= int(limited.headers["retry-after"]) <= 60
    split_lines = [("X-Forwarded-For", "10.9.9.9"), ("X-Forwarded-For", "198.51.100.7")]
    assert ask(client, "Who directed Jaws?", headers=split_lines).status_code == 429
    other_client = {"X-Forwarded-For": "198.51.100.7, 203.0.113.9"}
    assert ask(client, "Who directed Jaws?", headers=other_client).status_code == 200
    assert client.get("/health").status_code == 200  # only /ask is limited


def test_forwarded_header_is_ignored_unless_trusted(make_client):
    client = make_client(rate_limit_per_minute=2)
    responses = [ask(client, "Who directed Jaws?", headers={"X-Forwarded-For": f"203.0.113.{i}"}) for i in range(3)]
    assert [response.status_code for response in responses] == [200, 200, 429]


def test_metrics_are_hidden_by_default(client):
    assert client.get("/metrics").status_code == 404


def test_metrics_exposes_expected_names(make_client):
    client = make_client(metrics_public=True)
    ask(client, "Who directed Toy Story?")
    ask(client, "Who directed Oppenheimer (2023)?")
    text = client.get("/metrics").text
    for name in (
        "movie_rag_requests_total",
        "movie_rag_request_latency_seconds_bucket",
        "movie_rag_stage_latency_seconds_bucket",
        "movie_rag_refusals_total",
        "movie_rag_errors_total",
    ):
        assert name in text, name
    for stage in ("retrieve", "rerank", "generate"):
        assert f'movie_rag_stage_latency_seconds_count{{stage="{stage}"}}' in text
    assert 'movie_rag_requests_total{endpoint="/ask",status="200"}' in text


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert 'id="question"' in response.text and "Oppenheimer (2023)" in response.text


def test_cors_is_closed_by_default_and_opens_only_for_listed_origins(client, make_client):
    preflight = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    assert "access-control-allow-origin" not in client.options("/ask", headers=preflight).headers
    allowed = make_client(cors_origins="https://movies.example")
    response = allowed.options("/ask", headers={**preflight, "Origin": "https://movies.example"})
    assert response.headers["access-control-allow-origin"] == "https://movies.example"
    assert "access-control-allow-origin" not in allowed.options("/ask", headers=preflight).headers


def test_json_logs_are_structured():
    record = logging.LogRecord("movie_rag.api", logging.INFO, __file__, 1, "ask", None, None)
    record.fields = {"request_id": "r1", "latency_ms": 12.5, "refused": False}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["severity"] == "INFO" and payload["message"] == "ask"
    assert payload["request_id"] == "r1" and payload["latency_ms"] == 12.5


def test_api_key_never_reaches_logs_or_settings_repr(make_client, capsys):
    from movie_rag.config import Settings

    settings = Settings(groq_api_key="gsk_super_secret")
    assert "gsk_super_secret" not in repr(settings) and "gsk_super_secret" not in str(settings.model_dump())
    client = make_client(groq_api_key="gsk_super_secret")
    ask(client, "Who directed Toy Story?")
    assert "gsk_super_secret" not in capsys.readouterr().out
