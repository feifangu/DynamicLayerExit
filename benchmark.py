# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

import torch
import transformers
from tqdm import tqdm

from torchmetrics.text import BLEUScore, ROUGEScore, EditDistance
# TODO: create ExactMatch torchmetrics.text

from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.metric import Metric

from data import get_data, LowercaseProcessingFunction
from generate import load_model_and_tokenizer, setup
from utils import ROUGEScoreWrapper

import arguments
from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)

from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.dynamic_early_exit_generator import DynamicEarlyExitGenerationStrategy
from self_speculation.entropy_based_generator import EntropyExitGenerationStrategy
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
                avg = sum(generation_result.generation_strategy_result.exit_layers) / len(generation_result.generation_strategy_result.exit_layers)
                metric.update(torch.tensor(avg))

        # Add exit layer update
        for metric in self.avg_exit_layer.values():
            if generation_result.generation_strategy_result.exit_layers:
                avg = sum(generation_result.generation_strategy_result.exit_layers) / len(generation_result.generation_strategy_result.exit_layers)
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

def benchmark(
        model: torch.nn.Module, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        benchmark_arguments: BenchmarkArguments, 
        generation_config: GenerationConfig,
        seed = None,
    ):
    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    elif generation_config.generation_strategy == "dynamic_early_exit":
        generation_strategy: GenerationStrategy = DynamicEarlyExitGenerationStrategy()
    elif generation_config.generation_strategy == "entropy_exit":  # Add this
        generation_strategy: GenerationStrategy = EntropyExitGenerationStrategy()
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

    metric_result = metrics.compute()

    return metric_result


def main(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log arguments at beginning
    log.info(f"device={device}\n"
             "args={args}\n"
             "benchmark_arguments={benchmark_arguments}\n"
             "generation_config={generation_config}\n"
             "output_fname={output_fname}\n")

    # Setup and Run Benchmark
    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config)
    print(metric_result)

    # Save config and results to file
    with open(output_fname, "w") as f:
        json.dump(args.__dict__, f)
        json.dump(benchmark_arguments.__dict__, f)
        json.dump(generation_config.__dict__, f)
        json.dump(metric_result, f)

def process_cli_arguments() -> Tuple[arguments.Arguments, BenchmarkArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((arguments.Arguments, BenchmarkArguments, GenerationConfig))
    (
        general_arguments,
        benchmark_arguments,
        generation_config,
        _remaining,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_args = {}

    return general_arguments, benchmark_arguments, generation_config

if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    log.setLevel(level=logging.INFO) # TODO: set level based on argument
    main(args, benchmark_arguments, generation_config, f"{args.output_dir}/benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")