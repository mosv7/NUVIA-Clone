import pytest

from app.schemas.tasks import TaskRequest, validate_result


def test_compact_sentiment_never_guesses_from_invalid_or_missing_labels():
    from app.providers.qwen_interfaces import run_interface

    task = TaskRequest(schemaVersion=1, taskId="1", locale="en", text="Good. Bad.")
    values = iter(["mixed", "positive", "negative"])
    result = run_interface("sentiment", task, None, lambda *_: next(values))
    assert validate_result("sentiment", result, task)["label"] == "mixed"
    assert [item["label"] for item in result["sentences"]] == ["positive", "negative"]
    with pytest.raises(ValueError):
        run_interface("sentiment", task, None, lambda *_: "probably positive")


def test_aspect_predictions_keep_the_entire_source_sentence_as_literal_evidence():
    from app.providers.qwen import TAXONOMY
    from app.providers.qwen_interfaces import run_interface

    task = TaskRequest(schemaVersion=1, taskId="1", locale="en", text="Slow service.")
    labels = iter(["absent", "negative", "absent", "absent"])
    result = run_interface("aspects", task, TAXONOMY, lambda *_: next(labels))
    assert result["mentions"][0]["start"] == 0
    assert result["mentions"][0]["end"] == len(task.text)
    assert result["mentions"][0]["quote"] == task.text
    with pytest.raises(ValueError):
        run_interface(
            "aspects",
            task,
            TAXONOMY,
            lambda *_: "probably negative",
        )


def test_compact_assistant_keeps_abstention_and_rejects_unknown_fields():
    from app.providers.qwen_interfaces import run_interface

    task = TaskRequest(
        schemaVersion=1,
        taskId="1",
        locale="en",
        text="Question?",
        evidence=[{"id": "r1", "text": "Good service."}],
    )
    result = run_interface(
        "assistant",
        task,
        None,
        lambda *_: '{"answer":null,"sourceIds":[]}',
    )
    assert validate_result("assistant", result, task)["answer"] is None
    with pytest.raises(ValueError):
        run_interface(
            "assistant",
            task,
            None,
            lambda *_: '{"answer":null,"sourceIds":[],"extra":1}',
        )


@pytest.mark.parametrize(
    "capability,output",
    [
        ("assistant", '{"answer":"Slow service.","sourceIds":["invented"]}'),
        ("report", '{"title":"Service","summary":"Slow service.","sourceIds":["invented"]}'),
    ],
)
def test_compact_citations_cannot_invent_source_ids(capability, output):
    from app.providers.qwen_interfaces import run_interface

    task = TaskRequest(
        schemaVersion=1,
        taskId="1",
        locale="en",
        text="Question?",
        evidence=[{"id": "r1", "text": "Slow service."}],
    )
    with pytest.raises(ValueError):
        run_interface(capability, task, None, lambda *_: output)


def test_report_source_aliases_map_back_to_exact_authorized_original_ids():
    from app.providers.qwen_interfaces import run_interface

    task = TaskRequest(
        schemaVersion=1,
        taskId="1",
        locale="en",
        evidence=[{"id": "original-uuid", "text": "Slow service."}],
    )
    result = run_interface(
        "report",
        task,
        None,
        lambda *_: '{"title":"Service","summary":"Slow service.","sourceIds":[1]}',
    )
    assert result["evidence"] == [{"id": "original-uuid", "quote": "Slow service."}]
    assert result["sections"][0]["evidenceIds"] == ["original-uuid"]


def test_compact_survey_assigns_application_metadata_to_model_questions():
    from app.providers.qwen_interfaces import run_interface

    task = TaskRequest(schemaVersion=1, taskId="1", locale="en")
    result = run_interface(
        "survey_generation",
        task,
        None,
        lambda *_: (
            '{"title":"Service","description":"Feedback", "questions":'
            '["Speed?","Staff?","Ordering?","Payment?","Suggestions?"]}'
        ),
    )
    assert len(result["questions"]) == 5
    assert result["questions"][0]["type"] == "long_text"
    assert result["questions"][0]["label"] == "Speed?"
