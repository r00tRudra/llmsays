from unittest.mock import MagicMock, patch

import pytest

import llmsays as module


def _all_model_cases():
	for tier, provider_models in module.MODELS.items():
		for provider_name, expected_model in provider_models.items():
			yield pytest.param(tier, provider_name, expected_model, id=f"{tier}-{provider_name}")


@pytest.mark.parametrize("tier,provider_name,expected_model", list(_all_model_cases()))
def test_call_provider_uses_expected_model_slug(tier, provider_name, expected_model):
	mock_client = MagicMock()
	mock_response = MagicMock()
	mock_response.choices[0].message.content = "  mock output  "
	mock_client.chat.completions.create.return_value = mock_response

	with patch("llmsays._get_client", return_value=mock_client), patch("llmsays._record_latency"):
		output = module._call_provider(
			provider_name,
			tier,
			query_text="test query",
			max_tokens=32,
			temperature=0.2,
		)

	assert output == "mock output"
	mock_client.chat.completions.create.assert_called_once_with(
		model=expected_model,
		messages=[{"role": "user", "content": "test query"}],
		max_tokens=32,
		temperature=0.2,
	)


def test_all_tiers_have_all_providers_and_match_provider_catalog():
	providers = set(module.PROVIDERS.keys())
	for tier, provider_models in module.MODELS.items():
		assert set(provider_models.keys()) == providers, (
			f"Tier '{tier}' providers do not match PROVIDERS keys"
		)


def test_free_models_cover_exactly_the_mapped_models_for_each_provider():
	for provider_name in module.PROVIDERS:
		expected_models = {module.MODELS[tier][provider_name] for tier in module.MODELS}
		free_models = set(module.FREE_MODELS[provider_name])
		assert free_models == expected_models, (
			f"FREE_MODELS for '{provider_name}' are out of sync with MODELS"
		)


def test_each_tier_has_one_model_per_provider_and_expected_total_count():
	expected_provider_count = len(module.PROVIDERS)
	total = 0
	for tier, provider_models in module.MODELS.items():
		assert len(provider_models) == expected_provider_count, (
			f"Tier '{tier}' should define exactly one model per provider"
		)
		total += len(provider_models)

	assert total == len(module.MODELS) * expected_provider_count


@pytest.mark.parametrize("tier", list(module.MODELS.keys()))
def test_llmsays_passes_routed_tier_to_provider_calls(tier):
	with patch("llmsays.get_tier", return_value=tier), patch(
		"llmsays._provider_order", return_value=["Groq"]
	), patch("llmsays._call_provider", return_value=f"ok:{tier}") as mock_call:
		response = module.llmsays("test prompt")

	assert response == f"ok:{tier}"
	mock_call.assert_called_once_with("Groq", tier, "test prompt", 1024, 0.1)


@pytest.mark.parametrize("tier", list(module.MODELS.keys()))
def test_llmsays_attempts_all_provider_models_for_each_tier(tier):
	providers = list(module.PROVIDERS.keys())
	call_order = []

	def _fail_for_all(provider_name, call_tier, *_args):
		call_order.append((provider_name, call_tier))
		raise RuntimeError(f"{provider_name} unavailable")

	with patch("llmsays.get_tier", return_value=tier), patch(
		"llmsays._provider_order", return_value=providers
	), patch("llmsays._call_provider", side_effect=_fail_for_all):
		with pytest.raises(RuntimeError, match="All providers failed"):
			module.llmsays("test prompt")

	assert [provider for provider, _ in call_order] == providers
	assert all(call_tier == tier for _, call_tier in call_order)
