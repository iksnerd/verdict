"""Loading a model, and asking it things, in one place.

`Backend` (see `backend.py`) is the seam the HTTP API talks through: one `DecideRequest` in, one
`DecideResponse` out. That is the right shape for serving and the wrong shape for scoring 360
states in a loop.

So there are two seams, on purpose, and this is the batch one. `MlxBackend` is a thin
translation layer over an `Engine`, and anything doing bulk work uses the `Engine` directly and
keeps raw dicts instead of paying for pydantic on every question.

The tokenizer loads lazily, because it is only needed for `clip` and a caller that already has
short states should not pay for it.
"""
from __future__ import annotations

from typing import Any

#: Pre-converted FP16 of `convaiinnovations/laya`, published by the port's author on Hugging Face.
#: The port itself is the `mizorewww/laya-mlx` project on GitHub; there is no model repo under that
#: name and that path 401s (FINDINGS §13).
DEFAULT_MODEL = "aac6fef/laya-mlx"

#: Clipping the state is the main latency lever (FINDINGS §11): 512 tokens costs 87 ms per question
#: against 36 ms at 128, and a request's intent is nearly always stated up front.
PROMPT_TOKEN_BUDGET = 128

#: `build_sequence` truncates every rendered option to this many tokens, silently. Not ours to
#: change (it is laya's format), but `Engine.option_overflow` will tell you when you cross it.
OPTION_TOKEN_CAP = 48

#: Characters below which a criterion cannot reach OPTION_TOKEN_CAP tokens, for English prose at
#: roughly 4 characters a token. Deliberately generous: this is a cheap pre-filter, so it must
#: never miss a real overflow, and a false positive costs only a second opinion.
OPTION_CHAR_HINT = 150


def maybe_truncated(questions: dict[str, Any]) -> list[str]:
    """Criteria long enough that they *might* cross the token cap. Pure Python, no tokenizer.

    `Engine.option_overflow` answers this exactly, but it needs the tokenizer, and importing
    `transformers` to print a warning costs 1.3 s against a 30 ms decision. Calling it on every
    `verdict decide` took the command from 30 ms to 2.8 s, which is a worse bug than the one the
    warning is about. So: character count first, and the exact check only when it could matter.
    """
    out = []
    for qid, q in questions.items():
        crit = q.get("criteria")
        # Name the option rather than number it. laya renders `noul` as [false, true], so a
        # positional index here would disagree with the one `option_overflow` reports, and two
        # warnings about the same option pointing at different numbers is worse than neither.
        labelled = crit.items() if isinstance(crit, dict) else enumerate(crit or [])
        for label, v in labelled:
            if len(str(v)) > OPTION_CHAR_HINT:
                out.append(f"{qid} option {label!r} is {len(str(v))} characters and may cross the "
                           f"{OPTION_TOKEN_CAP}-token cap, which truncates it silently; "
                           f"`verdict decide --check` reports the exact count")
    return out


def quantize(agent: Any, bits: int) -> None:
    """Quantize a loaded laya agent in place, exactly as FINDINGS §36 measured: group 64, every
    layer that supports it except the action head. `DecisionModel` casts its input to
    `act_head.layers[0].weight.dtype`, which a quantized layer reports as uint32."""
    import mlx.core as mx
    import mlx.nn as nn

    if agent._inference is not agent.model:
        raise ValueError("a compiled agent keeps its FP16 graph; quantize before compiling")
    nn.quantize(agent.model, group_size=64, bits=bits,
                class_predicate=lambda path, m: (not path.startswith("act_head")
                                                 and hasattr(m, "to_quantized")))
    mx.eval(agent.model.parameters())


class Engine:
    """A loaded checkpoint plus its tokenizer. Construct it with `load`."""

    def __init__(self, model_id: str, agent: Any = None, bits: int = 16, **load_options: Any):
        if bits != 16 and load_options.get("compile"):
            raise ValueError("bits and compile together: a compiled forward pass keeps the FP16 "
                             "graph, so the quantization would do nothing")
        self.model_id = model_id
        self.bits = bits
        self.load_options = load_options
        self._agent = agent
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return f"laya-mlx:{self.model_id}" + (f" q{self.bits}" if self.bits != 16 else "")

    @property
    def agent(self) -> Any:
        """The underlying `laya_mlx` agent. Imported here so the API starts without MLX installed."""
        if self._agent is None:
            import laya_mlx

            self._agent = laya_mlx.load(self.model_id, **self.load_options)
            if self.bits != 16:
                quantize(self._agent, self.bits)
        return self._agent

    @property
    def tokenizer(self) -> Any:
        """Loaded on first use; only `clip` and `option_overflow` need it.

        A local checkpoint keeps the tokenizer in a `tokenizer/` directory beside the weights; a
        Hub repo keeps it in a `tokenizer/` **subfolder**, which is a different argument. Joining
        the path worked for the local case and produced an invalid repo id for the other, so
        `Engine("aac6fef/laya-mlx")` — this module's own `DEFAULT_MODEL` — raised
        `Repo id must be in the form 'repo_name' or 'namespace/repo_name'` the moment anything
        clipped. Nothing caught it because every caller passes the local path.
        """
        if self._tokenizer is None:
            from pathlib import Path

            from transformers import AutoTokenizer

            local = Path(self.model_id) / "tokenizer"
            if local.is_dir():
                self._tokenizer = AutoTokenizer.from_pretrained(str(local))
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        return self._tokenizer

    def clip_state(self, state: Any, budget: int | None = PROMPT_TOKEN_BUDGET) -> Any:
        """`clip` for any state, structured or not. A state within budget comes back as it was,
        still structured; a longer one comes back as the first `budget` tokens of exactly the text
        laya would have read (`serialize_state` is `json.dumps(state, ensure_ascii=False)`).

        `decide` never clipped, and on 600-character commands that cost about 380 ms a call
        against 65 ms (FINDINGS §29). A falsy budget turns clipping off.

        It returns the very object it was given when nothing was cut, so `clipped is not state`
        is how a caller tells that the model will read only part of it (the `truncated` mark).
        """
        if not budget:
            return state
        if isinstance(state, str):
            return self.clip(state, budget)
        import json

        text = json.dumps(state, ensure_ascii=False)
        clipped = self.clip(text, budget)
        return state if clipped == text else clipped

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Raw answers, keyed by question id. No pydantic: bulk callers do not need it."""
        return self.run(state, questions)["answers"]

    def run(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """laya's whole result: `answers` and `usage` (the tokens it read)."""
        return self.agent.predict(state, questions)

    def option_overflow(self, questions: dict[str, Any]) -> list[str]:
        """Options the model will only partly see, because `build_sequence` truncates each to
        `OPTION_TOKEN_CAP` tokens without saying so.

        Worth surfacing rather than enforcing: a long criterion still works, it is just quietly
        cut mid-sentence, and the author is the only one who can tell whether the dropped clause
        mattered. A 61-token criterion loses its last 13 tokens with no error anywhere.
        """
        from laya_mlx.common import render_options

        kinds = {"choice": "choice", "score": "score", "noul": "noul"}
        out = []
        for qid, q in questions.items():
            kind = kinds.get(q.get("type"))
            if kind is None:
                continue
            crit = q.get("criteria")
            opts = render_options({"t": kind, "crit": crit})
            # render_options returns [false, true] for noul and criteria order otherwise; name
            # them so this agrees with `maybe_truncated`.
            names = (["false", "true"] if kind == "noul"
                     else list(crit) if isinstance(crit, dict) else list(range(len(opts))))
            for i, text in enumerate(opts):
                label = names[i] if i < len(names) else i
                n = len(self.tokenizer(" " + text, add_special_tokens=False)["input_ids"])
                if n > OPTION_TOKEN_CAP:
                    out.append(
                        f"{qid} option {label!r} is {n} tokens; the model sees the first "
                        f"{OPTION_TOKEN_CAP} and never reads: "
                        f"{self.tokenizer.decode(self.tokenizer(' ' + text, add_special_tokens=False)['input_ids'][OPTION_TOKEN_CAP:])[:60]!r}"
                    )
        return out

    def clip(self, prompt: str, budget: int = PROMPT_TOKEN_BUDGET) -> str:
        """The first `budget` tokens of `prompt`. See PROMPT_TOKEN_BUDGET for why this is a knob."""
        ids = self.tokenizer(prompt)["input_ids"]
        if len(ids) <= budget:
            return prompt
        return self.tokenizer.decode(ids[1 : budget + 1], skip_special_tokens=True)


def load(model_id: str = DEFAULT_MODEL, bits: int = 16, **load_options: Any) -> Engine:
    """Load a checkpoint.

    `load_options` go straight to `laya_mlx.load`: `dtype`, `batch_size`, and the opt-in speed
    options `compile`, `pad_to_multiple`, `cache_prompts`. None are set by default, because the
    port's own defaults are the validated ones. `pad_to_multiple` was checked to leave answers
    unchanged; `batch_size` was measured and does nothing for this workload, since it batches
    across calls rather than across the questions within one (FINDINGS §11).
    """
    return Engine(model_id, bits=bits, **load_options)
