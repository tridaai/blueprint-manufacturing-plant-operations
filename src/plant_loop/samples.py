"""Bundled sample cases: the seeded shift board.

Synthetic exceptions for the fictional Meridian Fabrication, Plant 2 —
one of each shape the loops exist for: a two-round convergence, a
safety stop, a blocked-on-parts escalation, a non-converging
ambiguity, an injection in an operator note, and plain one-round
work. Used by the demo, the console, and the API's /samples route.
"""

from __future__ import annotations

import json
from importlib import resources

from .schemas import ExceptionCase


def load_sample_cases() -> list[ExceptionCase]:
    text = resources.files(__package__).joinpath("data/sample_cases.json").read_text(encoding="utf-8")
    return [ExceptionCase.model_validate(item) for item in json.loads(text)]
