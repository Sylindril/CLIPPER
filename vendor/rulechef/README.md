# vendor/rulechef

`benchmark_banking77.py` is RuleChef's own BANKING77 benchmark script, copied
unmodified from upstream so the comparison runs RuleChef's protocol and metric
code rather than a re-implementation.

- Source: <https://github.com/KRLabsOrg/rulechef>, `benchmarks/benchmark_banking77.py`
- Retrieved: 2026-09-25, from `main`
- Licence: Apache-2.0 (KRLabs). Unmodified.

`scripts/run_rulechef_banking77.py` imports this copy and replaces only its
loader, so RuleChef's sampling, refinement loop, and scoring run as published.

RuleChef is not a dependency of this package. Run it in its own interpreter:

```bash
uv venv .rulechef-venv --python 3.12
VIRTUAL_ENV=.rulechef-venv uv pip install "rulechef[agentic]==0.2.2" datasets grex
```

and pass `--rulechef-python .rulechef-venv/bin/python` to `scripts/run_banking77.py run`.
The model is Gemini 3.1 Flash-Lite through Gemini's OpenAI-compatible endpoint,
so `GEMINI_API_KEY` must be set.
