# llmsays

One-line LLM calls with automatic prompt-tier routing and provider failover.

`llmsays` keeps usage simple:

```python
from llmsays import llmsays

response = llmsays("Explain quantum tunneling in simple words")
print(response)
```

## Why llmsays

- Single function API: `llmsays(prompt)`
- Smart routing with `sentence-transformers/paraphrase-MiniLM-L3-v2`
- Tier selection: `small`, `medium`, `large`, `extra_large`
- Provider failover: Groq, NIM, OpenRouter, Fireworks, Baseten
- Latency-aware provider ordering
- Optional parallel provider querying for faster first-response

## Installation

```bash
pip install llmsays
```

## Required Environment Variables

Set at least one provider key (multiple keys recommended for failover):

- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `NVIDIA_API_KEY`
- `FIREWORKSAI_API_KEY`
- `BASETEN_API_KEY`

Example:

```bash
export GROQ_API_KEY="your_key"
export OPENROUTER_API_KEY="your_key"
```

## Quick Start

```python
from llmsays import llmsays

user_prompt = input("Here goes your prompt: ")
print(llmsays(user_prompt))
```



## Advanced Usage

Choose provider order:

```python
from llmsays import llmsays

print(
	llmsays(
		"Analyze this legal clause",
		provider_preference=["Groq", "Openrouter", "fireworks-ai"],
	)
)
```

Enable parallel provider queries (returns first successful response):

```python
from llmsays import llmsays

print(
	llmsays(
		"Design a production-ready architecture with tradeoffs",
		use_multiprocessing=True,
	)
)
```

## CLI Usage

```bash
llmsays "Explain transformers in simple terms"
llmsays "Analyze this legal clause" --providers Groq Openrouter
llmsays "Summarize this API contract" --use-multiprocessing
```



## How Routing Works

1. Heuristic pre-filter estimates complexity quickly.
2. Semantic routing refines tier selection.
3. Selected tier maps to provider-specific model choices.
4. If one provider fails, the next provider is attempted automatically.

## Notes

- Requires Python `>=3.9`
- Internet connection is required to call provider APIs
- Responses depend on the configured provider/model availability

## License

MIT

