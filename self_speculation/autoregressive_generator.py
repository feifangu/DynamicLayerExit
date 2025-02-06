# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional

import pandas as pd

import torch
import torch.nn.functional as F

import transformers
from scipy.stats import entropy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import decode_next_token, forward, forward_early


def _cosine_similarity(p: torch.Tensor, q: torch.Tensor, eps=1e-9) -> float:
    denom = (p.norm(2) * q.norm(2)).item()
    if denom < eps:
        return 0.0
    return float((p.dot(q) / denom).item())


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    """
    Computes KL(p || q) = sum( p_i * [log(p_i) - log(q_i)] ), safely:
      1) Upcast to float32 to avoid half-precision underflow
      2) Clamp both p,q so none are exactly 0 -> log(0) => -inf
      3) Sum p_i log(p_i/q_i)
      4) If the final result is slightly negative from rounding, clamp to 0
    """
    # 1) cast
    p_32 = p.float()
    q_32 = q.float()

    # 2) clamp to avoid log(0)
    p_32 = p_32.clamp(min=eps)
    q_32 = q_32.clamp(min=eps)

    # 3) kl = sum( p_i * (log p_i - log q_i) )
    #    If p and q sum to 1 over vocab_size, this is the discrete KL(p||q).
    kl_tensor = p_32 * (torch.log(p_32) - torch.log(q_32))
    kl_val = kl_tensor.sum().item()

    # 4) Clip negative due to floating rounding
    if kl_val < 0.0:
        kl_val = 0.0

    return kl_val


class AutoRegressiveGenerationStrategy(GenerationStrategy):
    def __init__(self, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer

    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_id: int,
        generation_config: GenerationConfig,
        logits_processors: Optional[
            transformers.generation.logits_process.LogitsProcessorList
        ] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
        save_path: str = "output.xlsx",
    ) -> GenerationStrategyResult:
        """Variant of `generate` with inputs/outputs formatted as token_ids."""
        past_key_values = None

        input_ids: torch.Tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        exit_query_cache = None

        # We'll store arrays for each metric: [num_layers][steps].
        max_probs_per_layer = [] if generation_config.analysis else None
        entropy_per_layer = [] if generation_config.analysis else None
        cosine_per_layer = [] if generation_config.analysis else None
        kl_per_layer = [] if generation_config.analysis else None
        topk_prob_diff_per_layer = [] if generation_config.analysis else None
        most_likely_tokens_per_layer = [] if generation_config.analysis else None

        num_layers = None
        k_for_topk = 15

        for step in range(generation_config.max_steps):
            if generation_config.exit_layer > 0:
                model_output = forward_early(
                    model,
                    input_ids,
                    past_key_values,
                    generation_config.exit_layer,
                    exit_query_cache,
                )
            else:
                model_output = forward(
                    model,
                    input_ids,
                    past_key_values,
                    analysis=generation_config.analysis,
                )

            logits = model_output.logits
            if logits_processors:
                logits = logits_processors(input_ids, logits)
            past_key_values = model_output.past_key_values

            # If in analysis mode, collect the list of layer-wise max probabilities
            if generation_config.analysis and model_output.partial_probs is not None:
                if num_layers is None:
                    num_layers = len(model_output.partial_probs)
                    # Initialize all metric lists
                    max_probs_per_layer = [[] for _ in range(num_layers)]
                    entropy_per_layer = [[] for _ in range(num_layers)]
                    cosine_per_layer = [[] for _ in range(num_layers)]
                    kl_per_layer = [[] for _ in range(num_layers)]
                    topk_prob_diff_per_layer = [[] for _ in range(num_layers)]
                    most_likely_tokens_per_layer = [[] for _ in range(num_layers)]

                # partial_probs[i] => distribution for layer i
                # partial_logits[i] => logits for layer i
                for layer_i in range(num_layers):
                    p_i = model_output.partial_probs[layer_i]  # shape [vocab_size]
                    log_i = model_output.partial_logits[
                        layer_i
                    ]  # shape [1, vocab_size]
                    p_max = float(p_i.max().item())
                    max_probs_per_layer[layer_i].append(p_max)

                    # eps = 1e-9
                    # p_clamp = p_i.clamp(min=eps)
                    # ent_val = -(p_clamp * p_clamp.log()).sum().item()
                    p_i_cpu = p_i.detach().cpu().numpy()
                    ent_val = entropy(p_i_cpu)
                    entropy_per_layer[layer_i].append(float(ent_val))

                    # (3) most likely token: argmax + decode
                    top_idx = int(p_i.argmax().item())
                    if self.tokenizer:
                        top_token_str = self.tokenizer.decode([top_idx])
                    else:
                        top_token_str = f"<id:{top_idx}>"
                    most_likely_tokens_per_layer[layer_i].append(top_token_str)

                    if layer_i == 0:
                        cos_val = 0.0
                        kl_val = 0.0
                        topk_diff_val = 0.0
                    else:
                        # Cosine
                        prev_p = model_output.partial_probs[layer_i - 1]
                        cos_val = _cosine_similarity(p_i, prev_p)
                        # KL
                        kl_val = _kl_divergence(prev_p, p_i)

                        # top-k prob diff
                        prev_log = model_output.partial_logits[
                            layer_i - 1
                        ]  # shape [1, vocab_size]
                        current_log = log_i  # shape [1, vocab_size]

                        current_top_vals, _ = torch.topk(
                            current_log, k_for_topk, dim=-1
                        )
                        last_top_vals, _ = torch.topk(prev_log, k_for_topk, dim=-1)

                        # shape [1, k], do local softmax
                        current_probs_topk = torch.softmax(current_top_vals, dim=-1)
                        last_probs_topk = torch.softmax(last_top_vals, dim=-1)

                        # prob_diff ~ average absolute difference
                        topk_diff_val = (
                            torch.abs(current_probs_topk - last_probs_topk)
                            .mean()
                            .item()
                        )

                    cosine_per_layer[layer_i].append(cos_val)
                    kl_per_layer[layer_i].append(kl_val)
                    topk_prob_diff_per_layer[layer_i].append(topk_diff_val)

            next_token, _ = decode_next_token(
                logits=logits,
                token_idx=-1,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )
            if streamer:
                streamer.put(next_token)
            next_token = next_token.item()
            if next_token == eos_token_id:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
            output_ids.append(next_token)
            # Don't concatenate `next_token` to original `input_ids` since we're using
            # the KV cache (`past_key_values`) to speed up generation.
            input_ids = torch.tensor([[next_token]]).to(input_ids)

        # Gather final data
        analysis_data = None
        if generation_config.analysis and num_layers is not None:
            analysis_data = {
                "max_prob": max_probs_per_layer,
                "entropy": entropy_per_layer,
                "cosine": cosine_per_layer,
                "kl_div": kl_per_layer,
                "topk_prob_diff": topk_prob_diff_per_layer,
                "most_likely_token": most_likely_tokens_per_layer,
            }

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,
            analysis_data=analysis_data if generation_config.analysis else None,
        )
