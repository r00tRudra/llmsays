---
title: 'llmsays: One-Line LLM Inference with Automatic Prompt-Tier Routing and Multi-Provider Failover'
tags:
  - Python
  - large language models
  - inference routing
  - provider failover
  - prompt complexity
  - sentence transformers
  - LLM abstraction
authors:
  - name: Abhiraj Adhikary
    orcid: 0009-0007-0009-4878
    affiliation: 1
  - name: Anik Chand
    orcid: 0009-0001-9720-4274
    affiliation: 2
  - name: Rudra Prasad Bhowmick
    orcid: 0009-0006-4759-7600
    affiliation: 3
affiliations:
  - name: Department of Data Science, Haldia Institute of Technology, India
    index: 1
  - name: Department of Computer Science and Engineering, Haldia Institute of Technology, India
    index: 2
  - name: Department of Information Technology, Haldia Institute of Technology, India
    index: 3
date: 2026-04-24
bibliography: paper.bib
doi: 10.5281/zenodo.19365666
figshare_doi: 10.6084/m9.figshare.31916274
---

# Summary

`llmsays` is an open-source Python library (MIT License) available at <https://github.com/abhirajadhikary06/llmsays> that reduces large language model (LLM) inference to a single function call. It introduces a two-stage hybrid routing mechanism that classifies an input prompt into one of four complexity tiers — `small`, `medium`, `large`, and `extra_large` — using a lightweight sentence-transformer model [@reimers2019], and subsequently dispatches the query to the most appropriate provider-specific model. The library supports five commercial inference providers (Groq [@groq2024], NVIDIA NIM [@nvidia2024], OpenRouter [@openrouter], Fireworks AI [@fireworks], and Baseten [@baseten]) with automatic latency-aware failover, eliminating manual provider management for researchers and practitioners alike. An optional multiprocessing mode enables parallel provider queries, returning the first successful response for latency-critical applications.

# Statement of Need

## Who Benefits

`llmsays` is designed for three primary audiences:

**Researchers and academics** who need to run LLM experiments or evaluations across many prompts without investing engineering effort in provider management. A researcher prototyping a reasoning benchmark, for instance, should not need to write bespoke retry logic or manually tune model selection for each run.

**Application developers** building LLM-powered products who want cost-efficient inference without hard-coding provider-specific logic. By automatically routing simple prompts to smaller, cheaper models and reserving larger models only for complex queries, `llmsays` reduces inference cost without sacrificing quality.

**Students and practitioners** entering the LLM space who face a steep learning curve just to make a first API call. `llmsays` removes that barrier entirely — a working LLM query requires no knowledge of provider SDKs, model names, or authentication patterns beyond setting an environment variable.

## Why Existing Tools Are Insufficient

The contemporary LLM ecosystem presents several interoperability challenges that existing tools do not fully resolve:

**API Fragmentation.** Each provider (Groq, NVIDIA NIM, OpenRouter, etc.) exposes an incompatible authentication scheme, base URL, and SDK, necessitating provider-specific wrappers. Developers maintaining multi-provider integrations must track breaking changes across all of them simultaneously.

**Model Selection Complexity.** Selecting a model commensurate with a task's complexity requires domain expertise; over-provisioning wastes compute budget, while under-provisioning degrades response quality. This decision is currently left entirely to the caller in every major abstraction library.

**Reliability.** Commercial APIs are subject to rate limits, transient outages, and quota exhaustion. Robust applications must implement retry and failover logic, which is non-trivial to do correctly and is typically re-implemented from scratch in each project.

Existing abstraction layers such as LiteLLM [@litellm] provide broad model coverage but require the caller to explicitly name the model on every invocation. `llmsays` goes one step further by automating model selection through prompt-complexity routing, enabling fully hands-free LLM inference for standard use-cases. No existing library combines automatic prompt-tier routing, latency-aware failover ordering, and optional parallel querying in a single zero-configuration interface.

# State of the Field

Several Python libraries provide abstractions over LLM inference APIs, but none combine prompt-complexity routing with multi-provider failover in a zero-configuration interface.

**LiteLLM** [@litellm] is the most widely adopted abstraction layer. It unifies the calling convention across over 100 providers using an OpenAI-compatible interface, handles retries, and supports cost tracking. However, it requires the caller to explicitly specify a model name on every invocation — model selection remains a manual responsibility. `llmsays` automates this step entirely.

**LangChain** is a compositional framework for building LLM-powered pipelines [@langchain]. Its `LLMChain` and `ChatModel` abstractions support multiple providers but are designed for complex agentic workflows rather than single-call inference. For users who only need to query a model, LangChain introduces significant boilerplate.

Neither library implements latency-aware provider ordering derived from runtime EWMA measurements, nor do they offer a parallel multi-provider query mode that races providers and returns the first successful response. `llmsays` is designed specifically for the common case — a single prompt, a single response — and makes that case require zero configuration beyond an API key.

The design philosophy of `llmsays` is therefore complementary rather than competitive: it occupies the lowest-friction point of the inference stack, and users who outgrow its abstractions can migrate to LiteLLM or LangChain without changing their application logic.

# Software Design

## Design Trade-offs

Three principal trade-offs shaped the architecture of `llmsays`:

**Accuracy vs. latency in routing.** A single large classification model would yield higher tier-assignment accuracy but would dominate the total inference latency for short prompts. The two-stage design — a near-zero-cost heuristic pre-filter followed by a 17M-parameter semantic model — was chosen to keep routing overhead below 50 ms on commodity CPU hardware in the common case, while preserving semantic accuracy for ambiguous prompts where surface-level heuristics fail.

**Breadth vs. depth of provider coverage.** Rather than wrapping every available provider, `llmsays` supports five providers selected for complementary model coverage across the tier matrix: Groq for low-latency small models, NVIDIA NIM for enterprise-grade large models, and OpenRouter, Fireworks AI, and Baseten for diverse mid-range and frontier options. This constrained set allows the maintainers to validate the model matrix against each provider's actual API behaviour rather than relying on untested community contributions.

**Simplicity vs. configurability.** The primary interface exposes a single function with sensible defaults. Advanced options (`provider_preference`, `use_multiprocessing`) are available but not required. This reflects a deliberate choice to optimise for the 80% use-case — a researcher or developer who wants a response without reading documentation — while not preventing power users from customising behaviour.

## Prompt-Tier Routing

The routing pipeline consists of two sequential stages designed for speed and accuracy.

**Stage 1 — Heuristic Pre-filter.** A lightweight rule-based filter examines surface-level features of the prompt (token count, presence of domain-specific keywords, structural complexity indicators such as multi-part questions or code snippets) to produce a coarse initial tier estimate at near-zero latency.

**Stage 2 — Semantic Refinement.** The prompt is encoded using `sentence-transformers/paraphrase-MiniLM-L3-v2` [@reimers2019; @sentencetransformers_pkg], a 17M-parameter bi-encoder optimised for fast CPU inference. The resulting embedding is compared against centroid representations of each tier's exemplar prompts via cosine similarity. If the semantic signal conflicts with the heuristic estimate, the semantic score takes precedence.

The result is one of four tiers:

| Tier | Intended Prompt Class |
|---|---|
| `small` | Simple factual queries, single-step tasks |
| `medium` | Multi-step reasoning, summarisation |
| `large` | Complex analysis, multi-document tasks |
| `extra_large` | Deep reasoning, large-context generation |

## Model Matrix

Each tier maps to a curated model per provider, as summarised below. The matrix is maintained in a versioned configuration file and can be extended by contributors as new models become available.

| Provider | small | medium | large | extra_large |
|---|---|---|---|---|
| Groq | gpt-oss-20b | qwen3-32b | llama-3.3-70b | gpt-oss-120b |
| NIM | nemotron-nano-9b | llama-3.3-nemotron-49b | nemotron-120b | llama-3.1-nemotron-253b |
| OpenRouter | step-3.5-flash | gemini-3-flash | deepseek-v3.2 | claude-opus-4.6 |
| Fireworks | qwen3-8b | qwen3-vl-30b | qwen3-vl-235b | qwen3-coder-480b |
| Baseten | gpt-oss-120b | MiniMax-M2.5 | Kimi-K2.5 | GLM-5 |

## Provider Failover and Latency Ordering

After tier selection, `llmsays` attempts providers in an order derived from an exponentially weighted moving average (EWMA) of observed response latencies, updated after each successful call within the process lifetime. If a provider raises a network error, authentication failure, or rate-limit exception, the library silently advances to the next candidate. This continues until a response is obtained or all configured providers are exhausted, in which case an informative exception is raised.

## Parallel Query Mode

For latency-critical workloads, the optional `use_multiprocessing=True` flag submits the request to all configured providers simultaneously using Python's `concurrent.futures.ProcessPoolExecutor`. The first successful response is returned and the remaining futures are cancelled. This trades additional API quota for reduced tail latency.

## Command-Line Interface

`llmsays` ships with a CLI entry-point for interactive and scripted use, supporting auto-routed queries, provider restriction, and parallel querying flags.

## Authentication

`llmsays` reads API credentials exclusively from environment variables, following the twelve-factor application convention [@12factor]. At least one key must be present; supplying multiple keys enables failover across: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `FIREWORKSAI_API_KEY`, and `BASETEN_API_KEY`.

# Usage

The minimal usage of `llmsays` requires no configuration beyond setting at least one provider API key as an environment variable:

```python
from llmsays import llmsays

user_prompt = input("Here goes your prompt: ")
response = llmsays(user_prompt)
print(response)
```

`llmsays()` decides the prompt tier using a hybrid router powered by `sentence-transformers/paraphrase-MiniLM-L3-v2` [@reimers2019], then picks the mapped model for each provider and dispatches the request with automatic latency-aware failover. No model name, provider name, or additional configuration is required from the caller.

# Architecture Overview

The internal control flow of a single `llmsays()` invocation is as follows:

1. **Input validation** — Prompt is type-checked and stripped of extraneous whitespace.
2. **Heuristic pre-filter** — Coarse tier estimate produced from lexical features.
3. **Semantic routing** — MiniLM embedding compared against tier centroids; final tier assigned.
4. **Provider ordering** — EWMA latency scores determine provider priority queue.
5. **Dispatch loop** — Providers are attempted sequentially (or in parallel); first success returned.
6. **Latency update** — EWMA table updated for the successful provider.

The library depends on `sentence-transformers` [@sentencetransformers_pkg] for the embedding model, `semantic-router` for hybrid routing support, and the official provider SDKs (`openai`, `groq`, `openrouter`, `fireworks-ai`, `baseten`) for inference dispatch. Standard-library modules handle multiprocessing and environment variable management.

# Installation

`llmsays` (current version: `0.1.3`) requires Python ≥ 3.9 and is distributed via PyPI:

```bash
pip install llmsays
```

The `sentence-transformers` model weights are downloaded automatically on first invocation and cached locally via HuggingFace Hub. The source code and issue tracker are available at <https://github.com/abhirajadhikary06/llmsays> under the MIT License.

# Testing

The test suite is executed with `pytest` from the repository root. Tests cover the routing logic (unit tests with mocked embeddings), the failover mechanism (simulated provider errors), and CLI argument parsing. Contributions are expected to maintain or improve coverage.

# Research Impact Statement

`llmsays` was developed to lower the barrier to reproducible LLM-based research. By eliminating the provider selection and authentication boilerplate that typically precedes any LLM experiment, it enables researchers to focus on experimental design rather than infrastructure. The library is publicly available on PyPI and GitHub under the MIT License, making it freely usable in both academic and commercial contexts.

The package has been archived on Zenodo (DOI: [10.5281/zenodo.19365666](https://doi.org/10.5281/zenodo.19365666)) and Figshare (DOI: [10.6084/m9.figshare.31916274](https://doi.org/10.6084/m9.figshare.31916274)) to ensure long-term citability and reproducibility. The versioned model matrix and open contribution model allow the community to extend provider and model coverage as the LLM ecosystem evolves.

Near-term impact is anticipated in research settings where multi-provider redundancy is operationally important — for example, large-scale prompt evaluation studies where a single provider outage would otherwise halt data collection. The parallel query mode directly addresses this failure mode.

# AI Usage Disclosure
 
Claude (Anthropic) was used as a writing assistance tool during the authoring of this paper. Its use was limited to drafting, paraphrasing, and improving the clarity of written prose. All AI-assisted content was reviewed, edited, and verified for accuracy and correctness by the authors. The software itself, its architecture, and its implementation were developed independently by the authors without the use of AI code generation tools. The authors take full responsibility for the accuracy of all content in this paper and the associated software.
 
# Acknowledgements

The authors thank Haldia Institute of Technology for institutional support and the open-source community whose tooling — in particular the Hugging Face ecosystem and the `sentence-transformers` library — made this work possible.

# References
