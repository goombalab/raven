# -*- coding: utf-8 -*-

from __future__ import annotations

import raven  # noqa: registers models with transformers
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model('raven')
class RavenLMWrapper(HFLM):
    def __init__(self, **kwargs) -> RavenLMWrapper:
        super().__init__(**kwargs)


if __name__ == "__main__":
    cli_evaluate()
