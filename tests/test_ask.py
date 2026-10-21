"""第1回: app/ask.py のテスト（LLMもAPIキーも使わない）。"""

from types import SimpleNamespace

from app import ask as ask_module


def _mock_client(monkeypatch, response):
    """genai.Client と get_api_key を差し替え、ダミーの response を返すようにする。"""
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "dummy-key")
    dummy_models = SimpleNamespace(generate_content=lambda **kwargs: response)
    dummy_client = SimpleNamespace(models=dummy_models)
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: dummy_client)


def test_ask_prints_token_counts_and_returns_text(monkeypatch, capsys):
    response = SimpleNamespace(
        text="ダミー回答",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
    )
    _mock_client(monkeypatch, response)

    result = ask_module.ask("テスト質問")

    captured = capsys.readouterr().out
    assert result == "ダミー回答"
    assert "12" in captured
    assert "34" in captured


def test_ask_handles_missing_usage_metadata(monkeypatch, capsys):
    response = SimpleNamespace(text="メタデータなし回答", usage_metadata=None)
    _mock_client(monkeypatch, response)

    result = ask_module.ask("テスト質問")

    captured = capsys.readouterr().out
    assert result == "メタデータなし回答"
    assert "[tokens]" not in captured
    assert "[time]" in captured


def test_ask_prints_elapsed_time(monkeypatch, capsys):
    """1問1答モードで応答時間が小数点以下2桁の秒数で表示されること。"""
    response = SimpleNamespace(
        text="ダミー回答",
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
    )
    _mock_client(monkeypatch, response)
    ticks = iter([100.0, 101.234])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(ticks))

    ask_module.ask("テスト質問")

    captured = capsys.readouterr().out
    assert "[time] 1.23秒" in captured


def test_chat_loop_keeps_session_and_prints_tokens(monkeypatch, capsys):
    """同じ chat セッションに対して複数の質問を順番に送り、exit で終了できること。"""
    sent_messages = []
    responses = iter(
        [
            SimpleNamespace(
                text="1回目の回答",
                usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
            ),
            SimpleNamespace(
                text="2回目の回答",
                usage_metadata=SimpleNamespace(prompt_token_count=45, candidates_token_count=30),
            ),
        ]
    )

    def fake_send_message(msg):
        sent_messages.append(msg)
        return next(responses)

    dummy_chat = SimpleNamespace(send_message=fake_send_message)
    dummy_chats = SimpleNamespace(create=lambda **kwargs: dummy_chat)
    dummy_client = SimpleNamespace(chats=dummy_chats)

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "dummy-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: dummy_client)

    inputs = iter(["最初の質問", "", "続きの質問", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    ask_module.chat_loop()

    captured = capsys.readouterr().out
    assert sent_messages == ["最初の質問", "続きの質問"]
    assert "1回目の回答" in captured
    assert "2回目の回答" in captured
    assert "入力: 10, 出力: 20" in captured
    assert "入力: 45, 出力: 30" in captured
    assert "終了します。" in captured


def test_chat_loop_exits_on_eof_or_keyboard_interrupt(monkeypatch, capsys):
    """Ctrl+D (EOFError) や Ctrl+C (KeyboardInterrupt) で安全に終了できること。"""
    dummy_chat = SimpleNamespace(send_message=lambda msg: None)
    dummy_chats = SimpleNamespace(create=lambda **kwargs: dummy_chat)
    dummy_client = SimpleNamespace(chats=dummy_chats)

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "dummy-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: dummy_client)

    for exc in (EOFError, KeyboardInterrupt):
        def raise_exc(_):
            raise exc()

        monkeypatch.setattr("builtins.input", raise_exc)
        ask_module.chat_loop()
        captured = capsys.readouterr().out
        assert "終了します。" in captured


def test_main_chat_option_calls_chat_loop(monkeypatch):
    """--chat オプションを指定すると chat_loop が呼ばれること。"""
    called = []
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat"])
    monkeypatch.setattr(ask_module, "chat_loop", lambda: called.append(True))

    ask_module.main()

    assert called == [True]


def test_chat_loop_prints_elapsed_time(monkeypatch, capsys):
    """対話モードでも各ターンの応答時間が小数点以下2桁の秒数で表示されること。"""
    response = SimpleNamespace(
        text="対話回答",
        usage_metadata=SimpleNamespace(prompt_token_count=15, candidates_token_count=25),
    )
    dummy_chat = SimpleNamespace(send_message=lambda msg: response)
    dummy_chats = SimpleNamespace(create=lambda **kwargs: dummy_chat)
    dummy_client = SimpleNamespace(chats=dummy_chats)

    monkeypatch.setattr(ask_module, "get_api_key", lambda: "dummy-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: dummy_client)

    inputs = iter(["質問1", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    ticks = iter([50.0, 50.856])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(ticks))

    ask_module.chat_loop()

    captured = capsys.readouterr().out
    assert "[time] 0.86秒" in captured


