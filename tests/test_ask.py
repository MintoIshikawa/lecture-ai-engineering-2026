from types import SimpleNamespace

from app import ask as ask_module


def _fake_client(monkeypatch):
    calls = {}

    def generate_content(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(usage_metadata=None, text="answer")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    def create_client(api_key: str):
        assert api_key == "test-key"
        return client

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", create_client)
    return calls


def test_ask_passes_temperature_in_generation_config(monkeypatch):
    calls = _fake_client(monkeypatch)

    assert ask_module.ask("question", temperature=0.4) == "answer"

    assert calls["config"].temperature == 0.4


def test_ask_uses_default_generation_config_when_temperature_is_omitted(monkeypatch):
    calls = _fake_client(monkeypatch)

    assert ask_module.ask("question") == "answer"

    assert calls["config"] is None


def test_main_passes_cli_temperature_to_ask(monkeypatch):
    calls = {}

    def fake_ask(question: str, temperature: float | None = None):
        calls["question"] = question
        calls["temperature"] = temperature
        return "answer"

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr("sys.argv", ["app.ask", "--temperature", "0.3", "question"])

    ask_module.main()

    assert calls == {"question": "question", "temperature": 0.3}
