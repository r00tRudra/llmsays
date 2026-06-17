"""
llmsays package: One-line LLM queries with prompt-tier routing and provider failover.
"""

import argparse
import os
import time
from multiprocessing.pool import ThreadPool
from typing import Dict, Iterable, List, Optional, Tuple
from openai import OpenAI
from .router import get_tier

_CLIENTS: Dict[str, OpenAI] = {}
_LATENCY_MS: Dict[str, Dict[str, float]] = {}

PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "NIM": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NIVIDIA_API_KEY",
        "alt_env_key": "NVIDIA_API_KEY",
    },
    "Openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "Fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKSAI_API_KEY",
    },
    "Baseten": {
        "base_url": "https://inference.baseten.co/v1",
        "env_key": "BASETEN_API_KEY",
    },
}

MODELS = {
    "small": {
        "Groq": "openai/gpt-oss-20b",
        "NIM": "nvidia/nvidia-nemotron-nano-9b-v2",
        "Openrouter": "stepfun/step-3.5-flash",
        "Fireworks": "fireworks/models/qwen3-8b",
        "Baseten": "openai/gpt-oss-120b",
    },
    "medium": {
        "Groq": "qwen/qwen3-32b",
        "NIM": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "Openrouter": "google/gemini-3-flash-preview",
        "Fireworks": "fireworks/models/qwen3-vl-30b-a3b-instruct",
        "Baseten": "MiniMaxAI/MiniMax-M2.5",
    },
    "large": {
        "Groq": "llama-3.3-70b-versatile",
        "NIM": "nvidia/nemotron-3-super-120b-a12b",
        "Openrouter": "deepseek/deepseek-v3.2-speciale",
        "Fireworks": "fireworks/models/qwen3-vl-235b-a22b-thinking",
        "Baseten": "moonshotai/Kimi-K2.5",
    },
    "extra_large": {
        "Groq": "openai/gpt-oss-120b",
        "NIM": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "Openrouter": "anthropic/claude-opus-4.6",
        "Fireworks": "fireworks/models/qwen3-coder-480b-a35b-instruct",
        "Baseten": "zai-org/GLM-5",
    },
}

FREE_MODELS = {
    "Groq": [
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ],
    "Openrouter": [
        "stepfun/step-3.5-flash",
        "google/gemini-3-flash-preview",
        "deepseek/deepseek-v3.2-speciale",
        "anthropic/claude-opus-4.6",
    ],
    "NIM": [
        "nvidia/nvidia-nemotron-nano-9b-v2",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ],
    "Fireworks": [
        "fireworks/models/qwen3-8b",
        "fireworks/models/qwen3-vl-30b-a3b-instruct",
        "fireworks/models/qwen3-vl-235b-a22b-thinking",
        "fireworks/models/qwen3-coder-480b-a35b-instruct",
    ],
    "Baseten": [
        "openai/gpt-oss-120b",
        "MiniMaxAI/MiniMax-M2.5",
        "moonshotai/Kimi-K2.5",
        "zai-org/GLM-5",
    ],
}


def _resolve_provider_key(provider_name: str) -> Optional[str]:
    provider = PROVIDERS[provider_name]
    key = os.getenv(provider["env_key"])
    alt_key_name = provider.get("alt_env_key")
    if not key and alt_key_name:
        key = os.getenv(alt_key_name)
    return key


def _provider_order(provider_preference: Optional[Iterable[str]] = None) -> List[str]:
    default_order = ["Groq", "NIM", "Openrouter", "Fireworks", "Baseten"]
    if not provider_preference:
        return default_order

    order: List[str] = []
    for name in provider_preference:
        for canonical in default_order:
            if canonical.lower() == str(name).strip().lower() and canonical not in order:
                order.append(canonical)
                break
    return order or default_order


def _latency_sorted_providers(tier: str, providers: List[str]) -> List[str]:
    tier_latencies = _LATENCY_MS.get(tier, {})
    known = [p for p in providers if p in tier_latencies]
    unknown = [p for p in providers if p not in tier_latencies]
    known_sorted = sorted(known, key=lambda p: tier_latencies[p])
    return known_sorted + unknown


def _get_client(provider_name: str) -> OpenAI:
    if provider_name in _CLIENTS:
        return _CLIENTS[provider_name]

    api_key = _resolve_provider_key(provider_name)
    if not api_key:
        raise ValueError(f"Missing API key for {provider_name}.")

    provider = PROVIDERS[provider_name]
    _CLIENTS[provider_name] = OpenAI(base_url=provider["base_url"], api_key=api_key)
    return _CLIENTS[provider_name]


def _record_latency(tier: str, provider_name: str, elapsed_ms: float) -> None:
    tier_latencies = _LATENCY_MS.setdefault(tier, {})
    previous = tier_latencies.get(provider_name)
    # Exponential moving average to stabilize provider ranking over time.
    tier_latencies[provider_name] = elapsed_ms if previous is None else (0.7 * previous + 0.3 * elapsed_ms)


def _call_provider(
    provider_name: str,
    tier: str,
    query_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    model_slug = MODELS[tier][provider_name]
    client = _get_client(provider_name)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_slug,
        messages=[{"role": "user", "content": query_text}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _record_latency(tier, provider_name, elapsed_ms)
    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content)


def _parallel_llmsays(
    providers: List[str],
    tier: str,
    query_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    errors: List[Tuple[str, str]] = []
    ordered = _latency_sorted_providers(tier, providers)

    with ThreadPool(processes=len(ordered)) as pool:
        async_results = {
            provider: pool.apply_async(
                _call_provider,
                (provider, tier, query_text, max_tokens, temperature),
            )
            for provider in ordered
        }
        while async_results:
            for provider in list(async_results.keys()):
                result = async_results[provider]
                if result.ready():
                    try:
                        text = result.get()
                        pool.terminate()
                        return text
                    except Exception as exc:
                        errors.append((provider, str(exc)))
                    del async_results[provider]
            time.sleep(0.01)

    failed = "; ".join(f"{name}: {err}" for name, err in errors)
    raise RuntimeError(
        "All providers failed for tier "
        f"'{tier}' in parallel mode. Configure valid API keys and endpoints. "
        f"Errors: {failed}"
    )


def benchmark_provider_latency(
    sample_query: str = "Say 'ok'.",
    tiers: Optional[Iterable[str]] = None,
    provider_preference: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    benchmark_tiers = list(tiers) if tiers else list(MODELS.keys())
    providers = _provider_order(provider_preference)
    output: Dict[str, Dict[str, float]] = {}

    for tier in benchmark_tiers:
        output[tier] = {}
        for provider_name in providers:
            try:
                _call_provider(provider_name, tier, sample_query, max_tokens=1, temperature=0.0)
                output[tier][provider_name] = _LATENCY_MS[tier][provider_name]
            except Exception:
                continue
    return output

def llmsays(
    query: object,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    provider_preference: Optional[Iterable[str]] = None,
    use_multiprocessing: bool = False,
) -> str:
    query_text = str(query).strip()
    if not query_text:
        raise ValueError("Prompt cannot be empty.")

    tier = get_tier(query_text)
    providers = _provider_order(provider_preference)

    if use_multiprocessing and providers:
        return _parallel_llmsays(providers, tier, query_text, max_tokens, temperature)

    provider_errors: List[Tuple[str, str]] = []
    for provider_name in _latency_sorted_providers(tier, providers):
        try:
            return _call_provider(provider_name, tier, query_text, max_tokens, temperature)
        except Exception as exc:
            provider_errors.append((provider_name, str(exc)))

    failed = "; ".join(f"{name}: {err}" for name, err in provider_errors)
    raise RuntimeError(
        "All providers failed for tier "
        f"'{tier}'. Configure at least one valid API key and reachable endpoint. "
        f"Errors: {failed}"
    )

def cli():
    parser = argparse.ArgumentParser(description="llmsays CLI")
    parser.add_argument("query", help="Prompt for LLM")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional provider order, e.g. --providers Groq Openrouter",
    )
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="Query providers in parallel and return the first successful response.",
    )
    args = parser.parse_args()
    print(
        llmsays(
            args.query,
            max_tokens=args.max_tokens,
            provider_preference=args.providers,
            use_multiprocessing=args.use_multiprocessing,
        )
    )

if __name__ == "__main__":
    cli()