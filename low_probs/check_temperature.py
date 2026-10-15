"""Can temperature be set to 0 for each sweep model, on the exact channel the sweep uses?

For each model (Jev excluded: it returns probabilities, not samples), sends one main-sweep
request (1% target, no repeat) three ways and reports whether the API accepts it:

  control       the request exactly as the sweep sends it (should succeed)
  temperature=0 the same request plus temperature 0
  temperature=1 the same request plus temperature 1 (separates "no temperature at all" from
                "no temperature other than 1")

OpenRouter silently drops parameters a provider doesn't support unless the request sets
provider.require_parameters, so every OpenRouter request here sets it: an unsupported temperature
then comes back as an error instead of being ignored. Anthropic's API (Fable 5) rejects
unsupported settings itself.

Being accepted only means the API took the parameter, not that it changes the output. 6 calls per
model, a few cents in total.

    python -m low_probs.check_temperature
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from harness.backends import AnthropicNativeBackend
from low_probs.experiment import (JEV, SWEEP_MODELS, make_backend, make_seeds, paper_hint,
                                  parse_choice, request_params, user_text)


def send(backend: Any, client: Any, params: Dict[str, Any]) -> str:
    if isinstance(backend, AnthropicNativeBackend):
        resp = client.messages.create(**params)
        return f"ok ({str(resp.content)[:80]})"
    resp = client.chat.completions.create(**params)
    text = resp.choices[0].message.content or ""
    return f"ok (answer {parse_choice(text)!r}, {text[:40]!r})"


def attempt(backend: Any, client: Any, model: str, text: str, temperature: Optional[float]) -> str:
    params = request_params(backend, model, text)
    if not isinstance(backend, AnthropicNativeBackend):
        params.setdefault("extra_body", {}).setdefault("provider", {})["require_parameters"] = True
    if temperature is not None:
        params["temperature"] = temperature
    try:
        return send(backend, client, params)
    except Exception as e:   # the point is to record the API's error, whatever it is
        return f"REJECTED: {type(e).__name__}: {str(e)[:300]}"


def main() -> None:
    text = user_text(paper_hint, make_seeds(1)[0], 0.01, 1)
    for model in SWEEP_MODELS:
        if model == JEV:
            continue
        backend = make_backend(model)
        client = backend.client()
        print(f"=== {model} ({type(backend).__name__}) ===")
        for label, t in (("control", None), ("temperature=0", 0.0), ("temperature=1", 1.0)):
            print(f"  {label:<14} {attempt(backend, client, model, text, t)}")
        print()


if __name__ == "__main__":
    main()
