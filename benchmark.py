# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import arguments

import torch
import transformers
import xlsxwriter
from arguments import Arguments, simple_parse_args_string

from data import get_data, LowercaseProcessingFunction
from generate import load_model_and_tokenizer, setup
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
# TODO: create ExactMatch torchmetrics.text

from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.metric import Metric

from torchmetrics.text import BLEUScore, EditDistance, ROUGEScore
from tqdm import tqdm
from utils import ROUGEScoreWrapper

log = logging.getLogger(__name__)


@dataclass
class BenchmarkArguments:
    dataset: str
    data_path: Optional[str] = None
    random_shuffle: bool = True
    num_samples: Optional[int] = None
    n_shot: Optional[int] = 0


@dataclass
class EvaluationExample:
    input: str
    output: str


@dataclass
class EvaluationMetrics:
    predicted_text: Dict[str, Metric]
    acceptance_rate: Dict[str, Metric]
    total_time: Dict[str, Metric]
    time_per_token: Dict[str, Metric]
    tokens_per_second: Dict[str, Metric]
    avg_exit_layer: Dict[str, Metric]

    def update(
        self,
        evaluation_example: EvaluationExample,
        generation_result: GenerationResult,
    ) -> None:
        if evaluation_example is not None:
            for metric in self.predicted_text.values():
                metric.update(
                    evaluation_example.output, generation_result.decoded_prediction
                )

        for metric in self.acceptance_rate.values():
            acceptance_rate = torch.tensor(
                generation_result.generation_strategy_result.acceptance_rate or -1
            )
            metric.update(acceptance_rate)

        for metric in self.total_time.values():
            metric.update(torch.tensor(generation_result.total_time))

        for metric in self.time_per_token.values():
            metric.update(torch.tensor(generation_result.time_per_token))

        for metric in self.tokens_per_second.values():
            metric.update(torch.tensor(generation_result.tokens_per_second))

        # Add exit layer update
        for metric in self.avg_exit_layer.values():
            if generation_result.generation_strategy_result.exit_layers:
                avg = sum(
                    generation_result.generation_strategy_result.exit_layers
                ) / len(generation_result.generation_strategy_result.exit_layers)
                metric.update(torch.tensor(avg))

    def compute(self) -> Dict[str, torch.Tensor]:
        return {
            "predicted_text": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.predicted_text.items()
            },
            "acceptance_rate": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.acceptance_rate.items()
            },
            "total_time": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.total_time.items()
            },
            "time_per_token": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.time_per_token.items()
            },
            "tokens_per_second": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.tokens_per_second.items()
            },
            "avg_exit_layer": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.avg_exit_layer.items()
            },
        }

    @classmethod
    def build_metrics(cls) -> "EvaluationMetrics":
        return cls(
            predicted_text={
                "rouge-l": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rougeL",
                        normalizer=LowercaseProcessingFunction,
                    )
                ),
                "rouge-1": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge1", normalizer=LowercaseProcessingFunction
                    )
                ),
                "rouge-2": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge2", normalizer=LowercaseProcessingFunction
                    )
                ),
                "rouge-3": ROUGEScoreWrapper(
                    ROUGEScore(
                        rouge_keys="rouge3", normalizer=LowercaseProcessingFunction
                    )
                ),
                "bleu_score": BLEUScore(
                    n_gram=4,
                ),
                "exact_match": EditDistance(),
            },
            acceptance_rate={"mean": Mean()},
            total_time={"mean": Mean()},
            time_per_token={"mean": Mean()},
            tokens_per_second={"mean": Mean()},
            avg_exit_layer={"mean": Mean()},
        )


def _accumulate_analysis_data(
    all_data: Dict[str, List[List[float]]],
    new_data: Dict[str, List[List[float]]],
    sample_idx: int,
):
    """
    Merge one sample's analysis_data into the global all_data structure.

    new_data[metric_name] is shape [num_layers][num_tokens].
      - new_data[metric_name][i] => list of length num_tokens for layer i

    We want all_data[metric_name] to become a big list of rows, each row shape:
      [layer_index, sample_idx, val_0, val_1, ... val_(num_tokens-1)]

    We'll store the numeric layer_index here, so we can do row labeling (e.g. "layer_003")
    at write time.
    """
    for metric_name, submatrix in new_data.items():
        # Ensure we have a list-of-rows in all_data for this metric
        if metric_name not in all_data:
            all_data[metric_name] = []
        # submatrix is shape [num_layers][num_tokens]
        num_layers = len(submatrix)
        for layer_i in range(num_layers):
            row_values = submatrix[layer_i]  # list of length num_tokens
            # Prepend [layer_i, sample_idx], then all the token values
            # We'll store them as float. For "most_likely_token", they're strings, but let's unify below.
            # Actually, "most_likely_token" is a list of strings, so let's handle that carefully:
            if (
                isinstance(row_values, list)
                and len(row_values) > 0
                and isinstance(row_values[0], str)
            ):
                # It's the "most_likely_token"
                # We'll store them in a list-of-strings row
                # e.g. [layer_i, sample_idx, "the", "world", ...]
                row = [layer_i, sample_idx] + row_values
            else:
                # It's numeric data
                row = [layer_i, sample_idx] + [float(x) for x in row_values]
            all_data[metric_name].append(row)


def _write_stacked_metric_sheet(
    workbook: xlsxwriter.Workbook,
    sheet_name: str,
    all_rows: List[List],
):
    """
    all_rows => a list of rows, each row shape: [layer_i, sample_idx, val0, val1, ..., valN].
    We'll produce an Excel sheet with:
      - row 0 => column headers: ["", "sample_idx", f"{sheet_name}_000", f"{sheet_name}_001", ...]
      - subsequent rows:
         col0 => "layer_{layer_i:03d}"
         col1 => sample_idx
         col2.. => the values
    """
    worksheet = workbook.add_worksheet(sheet_name)
    if not all_rows:
        return

    # We assume each row in all_rows has the same length
    # e.g. row = [layer_i, sample_idx, val0, val1, ..., valN]
    num_cols = len(all_rows[0])

    # The first 2 columns are "layer_i" and "sample_idx"
    # The rest are the token positions
    # => so we have (num_cols - 2) token positions
    num_tokens = num_cols - 2

    # row 0 => column headers
    # col0 => blank
    # col1 => "sample_idx"
    # col2.. => f"{sheet_name}_{000..}"
    worksheet.write(0, 0, "sample_idx")  # top-left corner
    worksheet.write(0, 1, "layer")
    for j in range(num_tokens):
        worksheet.write(0, j + 2, f"{sheet_name}_{j:03d}")

    # Now fill the data
    for row_idx, row_data in enumerate(all_rows, start=1):
        # row_data = [layer_i, sample_idx, val0, val1, ... valN]
        layer_i = row_data[0]
        sample_i = row_data[1]

        # col0 => row label = layer_{layer_i:03d}
        worksheet.write(row_idx, 0, sample_i)
        # col1 => sample_idx
        worksheet.write(row_idx, 1, f"layer_{layer_i:03d}")

        # For col2.. => the rest of the values
        for j in range(num_tokens):
            cell_val = row_data[2 + j]
            # If it's a string, xlsxwriter handles it; if float, it writes a number
            worksheet.write(row_idx, j + 2, cell_val)


def _save_analysis_to_excel_benchmark(
    filename: str, analysis_merged: Dict[str, List[List]]
):
    """
    Each key in analysis_merged is a metric: "max_prob", "entropy", ...
    The value is a big list-of-rows stacked across samples: [ [layer_i, sample_idx, val0, val1, ...], ... ]

    We'll produce one sheet per metric with the row/col headers described above.
    """
    if not analysis_merged:
        return

    workbook = xlsxwriter.Workbook(filename, {"nan_inf_to_errors": True})
    # We'll create a sheet for each metric
    # Must handle "most_likely_token" carefully if it has strings
    for metric_name in [
        "max_prob",
        "entropy",
        "cosine",
        "kl_div",
        "topk_prob_diff",
        "most_likely_token",
        "equal_final_token",
    ]:
        if metric_name not in analysis_merged:
            continue
        all_rows = analysis_merged[metric_name]
        _write_stacked_metric_sheet(workbook, metric_name, all_rows)

    workbook.close()


def benchmark(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizerBase,
    benchmark_arguments: BenchmarkArguments,
    generation_config: GenerationConfig,
    seed=None,
):
    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy(
            tokenizer=tokenizer
        )
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

    evaluation_set = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        n_shot=benchmark_arguments.n_shot,
        seed=seed,
        data_path=benchmark_arguments.data_path,
    )
    metrics = EvaluationMetrics.build_metrics()

    # We'll accumulate the layer-wise data across samples
    # e.g. analysis_merged["max_prob"] => big list of rows
    analysis_merged: Dict[str, List[List]] = {}

    for i, example in enumerate(tqdm(evaluation_set)):
        response: GenerationResult = generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        print(
            f"[Example]: {example.output}\n[Prediction]: {response.decoded_prediction}"
        )
        if response.num_tokens_generated == 0:
            print("Skipping empty generation")
            # TBD: print stats of emprty generations
            continue
        metrics.update(example, response)

        # If there's layer-wise analysis data, accumulate it
        if (
            generation_config.analysis
            and response.generation_strategy_result.analysis_data is not None
        ):
            _accumulate_analysis_data(
                analysis_merged,
                response.generation_strategy_result.analysis_data,
                sample_idx=i,
            )

    metric_result = metrics.compute()

    return metric_result, analysis_merged


def main(
    args: Arguments,
    benchmark_arguments: BenchmarkArguments,
    generation_config: GenerationConfig,
    output_fname: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log arguments at beginning
    log.info(
        f"device={device}\n"
        "args={args}\n"
        "benchmark_arguments={benchmark_arguments}\n"
        "generation_config={generation_config}\n"
        "output_fname={output_fname}\n"
    )

    # Setup and Run Benchmark
    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    metric_result, analysis_merged = benchmark(
        model, tokenizer, benchmark_arguments, generation_config
    )
    print(metric_result)

    # Save config and results to file
    with open(output_fname, "w") as f:
        json.dump(args.__dict__, f)
        json.dump(benchmark_arguments.__dict__, f)
        json.dump(generation_config.__dict__, f)
        json.dump(metric_result, f)

    # If we have analysis data from multiple samples, write it to Excel
    if generation_config.analysis and analysis_merged:
        excel_path = os.path.join(
            args.output_dir,
            f"benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        )
        _save_analysis_to_excel_benchmark(excel_path, analysis_merged)
        print(f"Layer-wise analysis for all samples saved to: {excel_path}")


def process_cli_arguments() -> (
    Tuple[arguments.Arguments, BenchmarkArguments, GenerationConfig]
):
    parser = transformers.HfArgumentParser(
        (arguments.Arguments, BenchmarkArguments, GenerationConfig)
    )
    (
        general_arguments,
        benchmark_arguments,
        generation_config,
        _remaining,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(
            general_arguments.model_args
        )
    else:
        general_arguments.model_args = {}

    return general_arguments, benchmark_arguments, generation_config


if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    log.setLevel(level=logging.INFO)  # TODO: set level based on argument
    main(
        args,
        benchmark_arguments,
        generation_config,
        f"{args.output_dir}/benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
