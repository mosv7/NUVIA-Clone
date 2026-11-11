"""Synthetic HTTP responses only; these tests never contact a paid provider."""

import json

import pytest

from tests.test_tasks import request


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVICE_KEY", "test-only-service-key")
    monkeypatch.setenv("AI_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("AI_MODEL_MANIFEST_DIR", str(tmp_path))
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "private-fixture-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSIONS", "768")
    return {"X-AI-Service-Key": "test-only-service-key"}


def test_explicit_cloud_selection_exposes_safe_readiness(configured):
    response = request("GET", "/api/v1/capabilities", headers=configured)
    entries = {item["capability"]: item for item in response.json()["capabilities"]}
    assert entries["sentiment"]["available"] is True
    assert entries["sentiment"]["provider"] == "gemini"
    assert entries["sentiment"]["modelVersion"].startswith("gemini/")
    assert entries["normalization"]["modelVersion"] == "builtin-1"
    assert entries["normalization"]["pipelineVersion"] == "nuvia-ai-1"
    assert entries["normalization"]["provider"] == "local"
    assert entries["clustering"]["available"] is True
    assert entries["clustering"]["provider"] == "gemini"
    assert entries["clustering"]["modelVersion"].startswith("gemini/")
    assert entries["forecasting"]["available"] is True
    assert entries["forecasting"]["provider"] == "gemini"
    assert entries["forecasting"]["modelVersion"].startswith("gemini/")
    assert "private-fixture-key" not in response.text


def test_cloud_dataset_capabilities_missing_key_stay_unavailable(configured, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    response = request("GET", "/api/v1/capabilities", headers=configured)
    entries = {item["capability"]: item for item in response.json()["capabilities"]}
    for capability in ("clustering", "forecasting"):
        assert entries[capability]["available"] is False
        assert entries[capability]["reason"] == "provider_not_configured"


def test_local_default_never_uses_key_without_explicit_selection(configured, monkeypatch):
    monkeypatch.delenv("AI_PROVIDER")
    response = request("GET", "/api/v1/capabilities", headers=configured)
    entry = next(x for x in response.json()["capabilities"] if x["capability"] == "sentiment")
    assert entry["available"] is False


@pytest.mark.parametrize("provider,key", [("gemini", ""), ("typo", "private-fixture-key")])
def test_bad_provider_configuration_never_falls_back(configured, monkeypatch, provider, key):
    monkeypatch.setenv("AI_PROVIDER", provider)
    monkeypatch.setenv("GEMINI_API_KEY", key)
    response = request(
        "POST",
        "/api/v1/tasks/sentiment",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "en", "text": "Good"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert "result" not in response.json()


def generated(value, *, finish="STOP"):
    return {
        "candidates": [
            {"finishReason": finish, "content": {"parts": [{"text": json.dumps(value)}]}}
        ]
    }


def inline_http(monkeypatch, response):
    from app.providers import gemini

    seen = []

    def post(config, method, payload):
        seen.append((method, payload))
        return response(payload) if callable(response) else response

    async def execute(config, path, task, **kwargs):
        return gemini.run_task(config, task)

    monkeypatch.setattr(gemini, "post_json", post)
    monkeypatch.setattr(gemini, "execute_isolated", execute)
    return seen


def test_structured_sentiment_passes_real_contract_validation(configured, monkeypatch):
    seen = inline_http(
        monkeypatch,
        generated({"label": "positive", "sentences": [{"sentenceId": "s1", "label": "positive"}]}),
    )
    response = request(
        "POST",
        "/api/v1/tasks/sentiment",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "ar", "text": "جيد"},
    )
    assert response.status_code == 200
    assert response.json()["pipelineVersion"] == "nuvia-gemini-json-5"
    assert response.json()["result"]["sentences"][0]["sentenceId"] == "s1"
    assert seen[0][0] == "generateContent"
    assert seen[0][1]["generationConfig"]["responseJsonSchema"]["type"] == "object"
    assert "جيد" in seen[0][1]["contents"][0]["parts"][0]["text"]


@pytest.mark.parametrize(
    "output",
    [
        generated({"label": "positive", "sentences": []}),
        generated({"label": "invented", "sentences": [{"sentenceId": "s1", "label": "positive"}]}),
        generated(
            {"label": "positive", "sentences": [{"sentenceId": "other", "label": "positive"}]}
        ),
        generated({}, finish="MAX_TOKENS"),
        {"promptFeedback": {"blockReason": "SAFETY"}},
    ],
)
def test_invalid_or_incomplete_generation_never_succeeds(configured, monkeypatch, output):
    inline_http(monkeypatch, output)
    response = request(
        "POST",
        "/api/v1/tasks/sentiment",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "en", "text": "Good"},
    )
    assert response.status_code == 502
    assert response.json()["status"] == "failed"
    assert "result" not in response.json()


def test_cloud_embeddings_preserve_ids_dimensions_and_normalize(configured, monkeypatch):
    # Characterize the observed Embedding 001 REST behavior: nested configuration is
    # ignored and produces the default 3072 dimensions; top-level fields select 768.
    def embedding_response(payload):
        return {
            "embeddings": [
                {"values": [3.0, 4.0] + [0.0] * (item.get("outputDimensionality", 3072) - 2)}
                for item in payload["requests"]
            ]
        }

    seen = inline_http(monkeypatch, embedding_response)
    response = request(
        "POST",
        "/api/v1/tasks/embeddings",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "t1",
            "locale": "en",
            "evidence": [{"id": "b", "text": "First"}, {"id": "a", "text": "Second"}],
        },
    )
    assert response.status_code == 200
    vectors = response.json()["result"]["vectors"]
    assert [v["id"] for v in vectors] == ["b", "a"]
    assert vectors[0]["values"][:2] == [0.6, 0.8]
    assert seen[0][0] == "batchEmbedContents"
    item = seen[0][1]["requests"][0]
    assert item["outputDimensionality"] == 768
    assert item["taskType"] == "CLUSTERING"
    assert "embedContentConfig" not in item


@pytest.mark.parametrize("vectors", [[], [{"values": [1.0]}], [{"values": [0.0] * 768}]])
def test_bad_embedding_cardinality_dimension_or_zero_norm_rejected(
    configured, monkeypatch, vectors
):
    inline_http(monkeypatch, {"embeddings": vectors})
    response = request(
        "POST",
        "/api/v1/tasks/embeddings",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "en", "text": "Good"},
    )
    assert response.status_code == 502
    assert "result" not in response.json()


@pytest.mark.parametrize("text", ["a" * 1025, "ج" * 513], ids=["ascii", "arabic"])
def test_cloud_embedding_long_source_is_rejected_before_remote_request(
    configured, monkeypatch, text
):
    def unexpected_remote(_payload):
        pytest.fail("Oversized source must never be sent to the provider")

    seen = inline_http(monkeypatch, unexpected_remote)
    response = request(
        "POST",
        "/api/v1/tasks/embeddings",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "en", "text": text},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "provider_input_too_large"
    assert seen == []


def test_cloud_embedding_rejects_entire_batch_if_one_source_is_oversized(configured, monkeypatch):
    def unexpected_remote(_payload):
        pytest.fail("No subset of an invalid batch may be sent to the provider")

    seen = inline_http(monkeypatch, unexpected_remote)
    response = request(
        "POST",
        "/api/v1/tasks/embeddings",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "r1",
            "locale": "en",
            "evidence": [{"id": "small", "text": "Good"}, {"id": "large", "text": "a" * 1025}],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "provider_input_too_large"
    assert seen == []


def test_cloud_embedding_boundary_sends_complete_arabic_source(configured, monkeypatch):
    text = "ج" * 512

    def embedding_response(payload):
        assert payload["requests"][0]["content"]["parts"][0]["text"] == text
        return {"embeddings": [{"values": [1.0] + [0.0] * 767}]}

    inline_http(monkeypatch, embedding_response)
    response = request(
        "POST",
        "/api/v1/tasks/embeddings",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "ar", "text": text},
    )
    assert response.status_code == 200
    assert response.json()["modelVersion"].endswith("/d768/norm1/b1024-v2")


def test_unverified_embedding_model_is_unavailable_without_disabling_text_tasks(
    configured, monkeypatch
):
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "some-unverified-model")
    response = request("GET", "/api/v1/capabilities", headers=configured)
    entries = {item["capability"]: item for item in response.json()["capabilities"]}
    assert entries["embeddings"]["available"] is False
    assert entries["embeddings"]["reason"] == "unsupported_embedding_model"
    assert entries["sentiment"]["available"] is True


def test_cloud_clustering_partitions_evidence_text_without_remote_vectors(configured, monkeypatch):
    seen = inline_http(
        monkeypatch,
        generated(
            {
                "clusters": [
                    {
                        "id": "c1",
                        "label": "Service speed",
                        "memberIds": ["a", "b"],
                        "evidenceIds": ["a", "b"],
                    },
                    {
                        "id": "c2",
                        "label": "Food quality",
                        "memberIds": ["c"],
                        "evidenceIds": ["c"],
                    },
                ],
                "noiseIds": [],
            }
        ),
    )
    response = request(
        "POST",
        "/api/v1/tasks/clustering",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "r1",
            "locale": "en",
            "evidence": [
                {"id": "a", "text": "Fast delivery."},
                {"id": "b", "text": "Quick service."},
                {"id": "c", "text": "Food was cold."},
            ],
            "parameters": {
                "vectors": [
                    {"id": "a", "values": [0.1, 0.2]},
                    {"id": "b", "values": [0.3, 0.4]},
                    {"id": "c", "values": [0.5, 0.6]},
                ],
                "sampleCount": 3,
                "totalResponses": 3,
            },
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert [cluster["id"] for cluster in result["clusters"]] == ["c1", "c2"]
    assert result["clusters"][0]["memberIds"] == ["a", "b"]
    assert result["noiseIds"] == []
    assert seen[0][0] == "generateContent"
    prompt = json.loads(seen[0][1]["contents"][0]["parts"][0]["text"])
    assert "vectors" not in prompt["parameters"]
    assert prompt["parameters"]["sampleCount"] == 3
    assert prompt["evidence"] == [
        {"id": "a", "text": "Fast delivery."},
        {"id": "b", "text": "Quick service."},
        {"id": "c", "text": "Food was cold."},
    ]


def test_cloud_clustering_requires_authorized_evidence(configured, monkeypatch):
    seen = inline_http(monkeypatch, generated({"clusters": [], "noiseIds": []}))
    response = request(
        "POST",
        "/api/v1/tasks/clustering",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "en"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_evidence"
    assert seen == []


@pytest.mark.parametrize(
    "clusters,noise",
    [
        ([{"id": "c1", "label": "Theme", "memberIds": ["a"], "evidenceIds": ["a"]}], []),
        (
            [{"id": "c1", "label": "Theme", "memberIds": ["a", "b"], "evidenceIds": ["a"]}],
            ["b", "b"],
        ),
        (
            [
                {
                    "id": "c1",
                    "label": "Theme",
                    "memberIds": ["a", "invented"],
                    "evidenceIds": ["a", "invented"],
                }
            ],
            [],
        ),
        (
            [
                {
                    "id": "c1",
                    "label": "Theme",
                    "memberIds": ["a", "b"],
                    "evidenceIds": ["a", "foreign"],
                }
            ],
            [],
        ),
    ],
    ids=["missing-member", "duplicate-noise", "invented-id", "foreign-evidence"],
)
def test_cloud_clustering_rejects_nonpartition_or_invented_output(
    configured, monkeypatch, clusters, noise
):
    inline_http(monkeypatch, generated({"clusters": clusters, "noiseIds": noise}))
    response = request(
        "POST",
        "/api/v1/tasks/clustering",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "r1",
            "locale": "en",
            "evidence": [{"id": "a", "text": "One."}, {"id": "b", "text": "Two."}],
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"
    assert "result" not in response.json()


def test_cloud_forecasting_projects_exactly_the_supplied_dates(configured, monkeypatch):
    observations = [
        {"date": f"2026-01-{day:02d}", "value": 5.0 + day / 100} for day in range(1, 31)
    ]
    seen = inline_http(
        monkeypatch,
        generated(
            {
                "points": [
                    {"date": "2026-02-01", "value": 5.5, "lower": 5.0, "upper": 6.0},
                    {"date": "2026-02-02", "value": 5.6, "lower": 5.1, "upper": 6.1},
                ],
                "sampleCount": 30,
                "uncertainty": "limited-history-projection",
            }
        ),
    )
    response = request(
        "POST",
        "/api/v1/tasks/forecasting",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "r1",
            "locale": "en",
            "parameters": {
                "observations": observations,
                "futureDates": ["2026-02-01", "2026-02-02"],
                "sampleCount": 30,
                "interval": "day",
                "metric": "weighted-mean-aspect-score",
            },
        },
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["sampleCount"] == 30
    assert [point["date"] for point in result["points"]] == ["2026-02-01", "2026-02-02"]
    assert seen[0][0] == "generateContent"
    prompt = json.loads(seen[0][1]["contents"][0]["parts"][0]["text"])
    assert prompt["parameters"]["observations"] == observations
    assert prompt["parameters"]["futureDates"] == ["2026-02-01", "2026-02-02"]


@pytest.mark.parametrize(
    "result",
    [
        {
            "points": [
                {"date": "2026-02-01", "value": 5.5},
                {"date": "2026-02-03", "value": 5.6},
            ],
            "sampleCount": 30,
        },
        {"points": [{"date": "2026-02-01", "value": 5.5}], "sampleCount": 30},
        {
            "points": [
                {"date": "2026-02-01", "value": 5.5},
                {"date": "2026-02-02", "value": 12.0, "lower": 11.0, "upper": 13.0},
            ],
            "sampleCount": 30,
        },
        {
            "points": [
                {"date": "2026-02-01", "value": 5.5},
                {"date": "2026-02-02", "value": 5.6, "lower": -1.0, "upper": 6.0},
            ],
            "sampleCount": 30,
        },
        {
            "points": [
                {"date": "2026-02-01", "value": 5.5},
                {"date": "2026-02-02", "value": 5.6},
            ],
            "sampleCount": 29,
        },
    ],
    ids=[
        "wrong-date",
        "missing-point",
        "value-out-of-range",
        "interval-out-of-range",
        "sample-count",
    ],
)
def test_cloud_forecasting_rejects_drifted_dates_ranges_or_counts(configured, monkeypatch, result):
    inline_http(monkeypatch, generated(result))
    response = request(
        "POST",
        "/api/v1/tasks/forecasting",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "r1",
            "locale": "en",
            "parameters": {
                "observations": [
                    {"date": f"2026-01-{day:02d}", "value": 5.0} for day in range(1, 31)
                ],
                "futureDates": ["2026-02-01", "2026-02-02"],
            },
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"
    assert "result" not in response.json()


def test_remote_schema_does_not_trigger_observed_maxitems_argument_rejection(
    configured, monkeypatch
):
    from app.services.runtime import ExecutionError

    def contains_array_maximum(value):
        if isinstance(value, dict):
            return "maxItems" in value or any(contains_array_maximum(v) for v in value.values())
        if isinstance(value, list):
            return any(contains_array_maximum(v) for v in value)
        return False

    def production_characterization(payload):
        # Controlled live comparison: identical model/prompt/minItems/schema with
        # maxItems present -> HTTP400 INVALID_ARGUMENT; removed -> HTTP200 STOP.
        schema = payload["generationConfig"]["responseJsonSchema"]
        if contains_array_maximum(schema):
            raise ExecutionError("provider_request_rejected")
        return generated(
            {"label": "positive", "sentences": [{"sentenceId": "s1", "label": "positive"}]}
        )

    inline_http(monkeypatch, production_characterization)
    response = request(
        "POST",
        "/api/v1/tasks/sentiment",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "ar", "text": "جيد"},
    )
    assert response.status_code == 200


def test_relaxed_remote_schema_retains_local_array_bounds(configured, monkeypatch):
    sections = [{"title": "Section", "text": "Good", "evidenceIds": ["e1"]}] * 31
    inline_http(
        monkeypatch,
        generated(
            {
                "title": "Report",
                "summary": "Good",
                "sections": sections,
                "evidence": [{"id": "e1", "quote": "Good"}],
            }
        ),
    )
    response = request(
        "POST",
        "/api/v1/tasks/report",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "r1",
            "locale": "en",
            "evidence": [{"id": "e1", "text": "Good"}],
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"


@pytest.mark.parametrize(
    "text,quote,start,end,expected_start,expected_end",
    [
        ("Delivery was quick and food was excellent.", "Delivery was quick", 0, 21, 0, 18),
        ("😀 الخدمة ممتازة.", "الخدمة ممتازة", 3, 16, 2, 15),
        ("Good service, good service.", "service", 19, 26, 19, 26),
    ],
)
def test_cloud_aspect_spans_derive_only_from_exact_unambiguous_sentence_quotes(
    configured, monkeypatch, text, quote, start, end, expected_start, expected_end
):
    inline_http(
        monkeypatch,
        generated(
            {
                "taxonomyVersion": "nuvia-gemini-aspects-v1",
                "mentions": [
                    {
                        "sentenceId": "s1",
                        "aspect": "staff",
                        "label": "positive",
                        "start": start,
                        "end": end,
                        "quote": quote,
                    }
                ],
            }
        ),
    )
    response = request(
        "POST",
        "/api/v1/tasks/aspects",
        headers=configured,
        json={"schemaVersion": 1, "taskId": "r1", "locale": "ar", "text": text},
    )
    assert response.status_code == 200
    mention = response.json()["result"]["mentions"][0]
    assert (mention["start"], mention["end"]) == (expected_start, expected_end)
    assert text[mention["start"] : mention["end"]] == quote


@pytest.mark.parametrize(
    "quote,sentence_id",
    [("service", "s1"), ("invented", "s1"), ("Other", "s1"), ("Other", "foreign")],
)
def test_cloud_aspect_span_repair_rejects_ambiguity_and_nonliteral_or_foreign_quotes(
    configured, monkeypatch, quote, sentence_id
):
    inline_http(
        monkeypatch,
        generated(
            {
                "taxonomyVersion": "nuvia-gemini-aspects-v1",
                "mentions": [
                    {
                        "sentenceId": sentence_id,
                        "aspect": "staff",
                        "label": "positive",
                        "start": 0,
                        "end": 1,
                        "quote": quote,
                    }
                ],
            }
        ),
    )
    response = request(
        "POST",
        "/api/v1/tasks/aspects",
        headers=configured,
        json={
            "schemaVersion": 1,
            "taskId": "r1",
            "locale": "en",
            "text": "Good service, good service. Other.",
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"


def test_aspect_alignment_is_idempotent_and_limited_to_declared_sentence():
    from app.providers.gemini import GeminiConfig, align_aspect_spans
    from app.schemas.tasks import TaskRequest, validate_result

    task = TaskRequest(
        schemaVersion=1, taskId="r1", locale="en", text="Good service. Good service."
    )
    output = {
        "taxonomyVersion": "nuvia-gemini-aspects-v1",
        "mentions": [
            {
                "sentenceId": "s2",
                "aspect": "staff",
                "label": "positive",
                "start": 0,
                "end": 1,
                "quote": "service",
            }
        ],
    }
    aligned = align_aspect_spans(output, task)
    assert aligned["mentions"][0]["start"] == 19
    assert aligned["mentions"][0]["end"] == 26
    assert align_aspect_spans(aligned, task) == aligned
    manifest = GeminiConfig(capability="aspects", key="fixture", model="fixture")
    assert validate_result("aspects", aligned, task, manifest) == aligned


def test_report_prompt_requests_prioritized_evidence_backed_actions_without_cluster_drift():
    from app.providers.gemini import GeminiConfig, instructions
    from app.schemas.tasks import TaskRequest

    config = GeminiConfig(capability="report", key="fixture", model="fixture")
    task = TaskRequest(schemaVersion=1, taskId="r1", locale="ar")
    prompt = instructions(config, task)
    assert "prioritized recommendations" in prompt
    assert "observations" in prompt and "hypotheses" in prompt
    assert "small sample" in prompt and "unknown causes" in prompt
    task.parameters = {"purpose": "cluster_summary"}
    cluster_prompt = instructions(config, task)
    assert "ONLY included cluster members" in cluster_prompt
    assert "prioritized recommendations" not in cluster_prompt


def test_provider_prompt_never_treats_platform_name_as_customer_business_identity():
    from app.providers.gemini import GeminiConfig, instructions
    from app.schemas.tasks import TaskRequest

    config = GeminiConfig(capability="report", key="fixture", model="fixture")
    for locale in ("ar", "en"):
        task = TaskRequest(schemaVersion=1, taskId="r1", locale=locale)
        prompt = instructions(config, task)
        assert "NUVIA is the software platform, not the reviewed business" in prompt
        assert "Never invent a business name" in prompt
        assert "Customer feedback report" in prompt
        assert "تقرير آراء العملاء" in prompt


@pytest.mark.parametrize(
    "capability",
    [
        "sentiment",
        "aspects",
        "assistant",
        "survey_generation",
        "clustering",
        "forecasting",
    ],
)
def test_generic_report_title_examples_do_not_leak_into_other_tasks(capability):
    from app.providers.gemini import GeminiConfig, instructions
    from app.schemas.tasks import TaskRequest

    config = GeminiConfig(capability=capability, key="fixture", model="fixture")
    task = TaskRequest(schemaVersion=1, taskId="r1", locale="ar")
    prompt = instructions(config, task)
    assert "Customer feedback report" not in prompt
    assert "تقرير آراء العملاء" not in prompt
    assert "Never invent a business name" in prompt
    if capability == "survey_generation":
        assert "survey-specific title" in prompt


def test_clustering_prompt_requires_exact_text_partition_with_localized_labels():
    from app.providers.gemini import GeminiConfig, instructions
    from app.schemas.tasks import TaskRequest

    config = GeminiConfig(capability="clustering", key="fixture", model="fixture")
    task = TaskRequest(schemaVersion=1, taskId="r1", locale="ar")
    prompt = instructions(config, task)
    assert "exactly once" in prompt
    assert "memberIds" in prompt and "noiseIds" in prompt
    assert "evidenceIds" in prompt
    assert "invented" in prompt
    assert "vectors" not in prompt


def test_forecasting_prompt_limits_projection_to_supplied_dates_and_metric_range():
    from app.providers.gemini import GeminiConfig, instructions
    from app.schemas.tasks import TaskRequest

    config = GeminiConfig(capability="forecasting", key="fixture", model="fixture")
    task = TaskRequest(schemaVersion=1, taskId="r1", locale="en")
    prompt = instructions(config, task)
    assert "futureDates" in prompt
    assert "sampleCount" in prompt
    assert "0 to 10" in prompt
    assert "lower" in prompt and "upper" in prompt
    assert "do not invent" in prompt
