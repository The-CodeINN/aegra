from aegra_api.services.career_advisor_activation import (
    extract_latest_human_text,
    is_career_roadmap_trigger,
)


def test_is_career_roadmap_trigger_accepts_expected_phrases() -> None:
    assert is_career_roadmap_trigger("Generate my career roadmap")
    assert is_career_roadmap_trigger("Generate my personalised career roadmap")
    assert is_career_roadmap_trigger("Can you build my career roadmap?")


def test_is_career_roadmap_trigger_rejects_non_trigger_messages() -> None:
    assert not is_career_roadmap_trigger("Hi there")
    assert not is_career_roadmap_trigger("What Python libraries should I learn?")
    assert not is_career_roadmap_trigger("")


def test_extract_latest_human_text_prefers_last_human_message() -> None:
    payload = {
        "messages": [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi"},
            {
                "type": "human",
                "content": [{"type": "text", "text": "Generate my personalised career roadmap"}],
            },
        ]
    }

    assert extract_latest_human_text(payload) == "Generate my personalised career roadmap"
