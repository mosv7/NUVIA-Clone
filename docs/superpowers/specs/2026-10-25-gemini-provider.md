# Optional Gemini provider

The user approved Gemini online integration to avoid model downloads, with an
explicit requirement to retain the local model implementation for future team
models. This extends the existing AI service rather than replacing it.

FastAPI selects `AI_PROVIDER=local` (default) or `gemini`. No local failure may
trigger a cloud call. Node authorization, persistence, job processing and public
DTOs remain unchanged. Builtin preprocessing stays local. Gemini handles supported
text tasks and embeddings through bounded HTTPS requests with private credentials,
structured responses and the existing semantic/evidence validators. Forecasting
and supported local model packages remain available through the local path.

Provider/model provenance distinguishes cloud results from local results. Missing
credentials, rate limits, timeouts and invalid outputs must be honest failures.
No secrets, private text or raw provider errors are returned or logged. Gemini
credentials are passed only to the AI development child, never the browser child.
Existing model files, manifests, weights, runtime extras and configuration are
preserved. No new model downloads or automatic provider activation.

Acceptance: local regressions continue passing; mocked provider transport verifies
all supported contracts and error/security boundaries; root launch/setup tests
verify preservation and credential isolation. Real API quality/quotas require a
user-supplied key and separate synthetic evaluation. No live acceptance claim
without actual calls. No UI redesign or production deployment is included.

## Extension approved 2026-11-11: every learned capability via Gemini

The user approved extending the same explicit selection so Gemini handles every
learned capability, adding `clustering` and `forecasting` to the original six
tasks. Deterministic `language`/`normalization` builtins stay local, and the local
package path plus the Qwen selection remain intact as explicit alternatives —
nothing is removed and there is still no implicit fallback between providers.

Clustering groups authorized evidence text semantically; the numeric `vectors`
parameter is not sent to the provider. The strict partition validator still
requires every supplied evidence ID exactly once across cluster members and
noise. Forecasting must project exactly the supplied futureDates in order, with
values and interval bounds inside the observed 0–10 metric range; drift fails as
`invalid_model_output` in both the service and Node validation. Clustering
without evidence returns `insufficient_evidence` (422) like the local path.

Provenance moves to `nuvia-gemini-json-5`. Mocked-transport regressions cover the
new contracts, payload stripping and error boundaries. No live scientific
acceptance is claimed for the added capabilities; semantic grouping quality and
predictive accuracy require separate evaluation with a real key.
