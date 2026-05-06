# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from raven.models.raven.configuration_raven import RavenConfig
from raven.models.raven.modeling_raven import RavenForCausalLM, RavenModel

AutoConfig.register(RavenConfig.model_type, RavenConfig, exist_ok=True)
AutoModel.register(RavenConfig, RavenModel, exist_ok=True)
AutoModelForCausalLM.register(RavenConfig, RavenForCausalLM, exist_ok=True)

__all__ = ['RavenConfig', 'RavenForCausalLM', 'RavenModel']
