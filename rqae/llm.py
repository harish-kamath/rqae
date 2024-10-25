from transformers import AutoModelForCausalLM, AutoConfig
from typing import Callable, Optional, Union
import torch
import torch.nn as nn


class AMCLM(nn.Module):
    def __init__(self, *args, layer: str = "half", **kwargs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        if layer == "half":
            layer = len(self.model.model.layers) // 2

        self.layer = layer
        self.extra_layers = None
        self.hooks = []

    def hook(self, hook: Callable):
        self.hooks.append(
            self.model.model.layers[self.layer - 1].register_forward_hook(hook)
        )

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def load_extra_layers(self, hook: Callable = None):
        if hook is not None:
            new_hook = self.model.model.layers[self.layer - 1].register_forward_hook(
                hook
            )
            self.hooks.append(new_hook)
        if self.extra_layers is None:
            return
        self.model.model.layers.extend(
            self.extra_layers(self.extra_layers.to(self.device))
        )
        self.extra_layers = None

    def deload_extra_layers(self):
        if self.extra_layers is not None:
            return
        self.__dict__["extra_layers"] = self.model.model.layers[self.layer :].to("cpu")
        self.model.model.layers = self.model.model.layers[: self.layer].to(
            self.model.model.device
        )
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)


class Gemma2(AMCLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def norm(self, hidden_states):
        return self.model.model.norm(hidden_states)

    def denorm(self, hidden_states, original_hidden_states):
        hidden_states = hidden_states / (1.0 + self.model.model.norm.weight.float())
        hidden_states = hidden_states.float() / torch.rsqrt(
            original_hidden_states.float().pow(2).mean(-1, keepdim=True) + 1e-6
        )
        return hidden_states
