# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import os
import random
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import colorama
import torch
import transformers
import xlsxwriter

from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.dynamic_early_exit_first_generator import (
    DynamicEarlyExitFirstGenerationStrategy,
)
from self_speculation.dynamic_early_exit_max_generator import (
    DynamicEarlyExitMaxGenerationStrategy,
)
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)
from self_speculation.self_speculation_generator import (
    SelfSpeculativeGenerationStrategy,
)
from self_speculation.speculative_streamer import SpeculativeTextStreamer


class StreamerType(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    SPECULATIVE = "speculative"


@dataclass
class GenerateArguments:
    streamer: StreamerType = StreamerType.STANDARD


def setup(args: Arguments, device: str = "cuda"):
    backend_str = "cpu:gloo" if "cpu" in device else "cuda:nccl,cpu:gloo"
    torch.distributed.init_process_group(
        backend=backend_str, timeout=datetime.timedelta(hours=48)
    )
    rank = int(os.environ["LOCAL_RANK"])

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if rank != 0:
        # only run on rank 0, we don't support parallel inference yet
        exit()


def load_model_and_tokenizer(args: Arguments, device: str = "auto"):
    local_model_path: str = args.model

    # initialize model
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        local_model_path,
        use_safetensors=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer


def _write_metric_sheet(
    workbook,
    sheet_name: str,
    metric_data: List[List[float]],
    # If you want row=layers, col=tokens, set this True
    # If you want row=tokens, col=layers, set this False
    row_layers_col_tokens: bool = False,
):
    """
    metric_data: shape [num_layers][num_steps].
      - metric_data[layer_i] = list of length = num_steps
    We create a sheet with an extra row/col as requested:
      - top-left cell is blank
      - first row after that: "layer_000", "layer_001", ...
      - first column after that: f"{sheet_name}_001", f"{sheet_name}_002", ...

    By default below: each row = token index, each column = layer index.
    If row_layers_col_tokens=True, we would transpose usage.
    """
    worksheet = workbook.add_worksheet(sheet_name)

    num_layers = len(metric_data)
    if num_layers == 0:
        return
    num_tokens = len(metric_data[0])

    # 1) Write column headers for the token positions in row=0, col=j+1
    #    e.g.  sheet_name_000, sheet_name_001, ...
    worksheet.write(0, 0, "")  # top-left corner empty
    for j in range(num_tokens):
        col_label = f"{sheet_name}_{j:03d}"
        worksheet.write(0, j + 1, col_label)

    # 2) For each row i => layer i
    #    First column => "layer_{i:03d}"
    #    Then fill columns with metric_data[i][j]
    for i in range(num_layers):
        row_label = f"layer_{i:03d}"
        worksheet.write(i + 1, 0, row_label)

        for j in range(num_tokens):
            val = metric_data[i][j]

            worksheet.write(i + 1, j + 1, val)


def save_analysis_to_excel(filename: str, analysis_data: dict):
    """
    analysis_data = {
      'max_prob': [[...], ...],
      'entropy':  [[...], ...],
      'cosine':   [[...], ...],
      'kl_div':   [[...], ...],
      'topk_prob_diff': [[...], ...],
      ...
    }
    """
    if not analysis_data:
        return
    workbook = xlsxwriter.Workbook(filename, {"nan_inf_to_errors": True})

    # We'll create a sheet for each metric
    for metric_name in ["max_prob", "entropy", "cosine", "kl_div", "topk_prob_diff"]:
        if metric_name not in analysis_data:
            continue
        metric_data = analysis_data[metric_name]
        _write_metric_sheet(workbook, metric_name, metric_data)

    workbook.close()


def main(
    args: Arguments,
    generate_arguments: GenerateArguments,
    generation_config: GenerationConfig,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    streamer = None
    match generate_arguments.streamer:
        case StreamerType.NONE:
            streamer = None
        case StreamerType.STANDARD:
            streamer = transformers.TextStreamer(tokenizer)
        case StreamerType.SPECULATIVE:
            streamer = SpeculativeTextStreamer(tokenizer)
        case _:
            raise ValueError(f"Unsupported streamer type {generate_arguments.streamer}")

    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    elif generation_config.generation_strategy == "dynamic_early_exit_first":
        generation_strategy: GenerationStrategy = (
            DynamicEarlyExitFirstGenerationStrategy()
        )
    elif generation_config.generation_strategy == "dynamic_early_exit_max":
        generation_strategy: GenerationStrategy = (
            DynamicEarlyExitMaxGenerationStrategy()
        )
    else:
        raise Exception(
            f"Unsupported generation strategy: {generation_config.generation_strategy}"
        )

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    # Warmup
    warmup = 1
    for _ in range(warmup):
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generate(
            **tokenizer("This is a warmup prompt", return_tensors="pt").to(device),
            max_new_tokens=10,
        )

    while True:
        print()
        # print("Enter a prompt and then press ctrl+d twice for the model to complete:")
        print("Enter a prompt for the model to complete:")
        print("======================================================================")
        print()

        print(colorama.Fore.BLUE, end="")
        prompt = sys.stdin.read()
        print(colorama.Style.RESET_ALL, end=" ")

        try:
            response: GenerationResult = generator.generate(
                prompt=prompt,
                generation_config=generation_config,
                streamer=streamer,
            )
        except:
            print(colorama.Style.RESET_ALL)
            traceback.print_exc()
            raise
        num_tokens = response.num_tokens_generated
        total_time = response.total_time

        if streamer:
            streamer.end()
        else:
            print(response.decoded_prediction)

        print(colorama.Style.RESET_ALL)
        print()
        print(f"\tTime taken: {total_time :.3f}s")
        print(f"\tNumber of tokens: {num_tokens}")
        print(f"\tTime per token: {total_time / num_tokens : .3f}s")
        print(f"\tTokens per second: {num_tokens / total_time :.3f}")
        if generation_config.generation_strategy in (
            "self_speculative",
            "dynamic_early_exit",
        ):
            print(
                f"\tAcceptance Rate: {response.generation_strategy_result.acceptance_rate:.2%}"
            )
        if generation_config.generation_strategy == "dynamic_early_exit":
            print(
                "\nExit layers used:", response.generation_strategy_result.exit_layers
            )
            print(
                f"Average exit layer: {sum(response.generation_strategy_result.exit_layers) / len(response.generation_strategy_result.exit_layers):.2f}"
            )
        print()

        # --- Save analysis to Excel if requested ---
        if (
            generation_config.analysis
            and generation_config.generation_strategy == "autoregressive"
            and response.generation_strategy_result.analysis_data is not None
        ):
            excel_path = os.path.join(
                args.output_dir,
                f"generate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            )
            save_analysis_to_excel(
                excel_path, response.generation_strategy_result.analysis_data
            )
            print(f"Analysis Excel saved to: {excel_path}")


def process_cli_arguments() -> Tuple[Arguments, GenerateArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser(
        (Arguments, GenerateArguments, GenerationConfig)
    )
    (
        general_arguments,
        generate_arguments,
        generation_config,
        _remaining,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(
            general_arguments.model_args
        )
    else:
        general_arguments.model_args = {}

    return general_arguments, generate_arguments, generation_config


if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    main(args, benchmark_arguments, generation_config)
