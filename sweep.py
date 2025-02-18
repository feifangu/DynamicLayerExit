# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import arguments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate
import torch
import transformers
from arguments import Arguments, simple_parse_args_string
from benchmark import (
    benchmark,
    BenchmarkArguments,
    load_model_and_tokenizer,
    process_cli_arguments,
    setup,
)
from scipy.interpolate import griddata
from self_speculation.generator_base import GenerationConfig


@dataclass
class SweepArguments:
    exit_layer_first: Optional[int] = 1
    exit_layer_last: Optional[int] = 15
    exit_layer_step: Optional[int] = 1
    num_speculations_first: Optional[int] = 1
    num_speculations_last: Optional[int] = 6
    num_speculations_step: Optional[int] = 1

    threshold_first: Optional[float] = 0.1
    threshold_last: Optional[float] = 0.9
    threshold_step: Optional[float] = 0.05
    min_layer_first: Optional[int] = 1
    min_layer_last: Optional[int] = 15
    min_layer_step: Optional[int] = 1


def sweep(
    args: Arguments,
    benchmark_arguments: BenchmarkArguments,
    generation_config: GenerationConfig,
    sweep_arguments: SweepArguments,
):
    results: List[Dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_fname = f"{args.output_dir}/sweep_{timestamp}.csv"
    pdf_fname = f"{args.output_dir}/sweep_{timestamp}.pdf"
    if generation_config.generation_strategy == "self_speculative":
        for exit_layer in range(
            sweep_arguments.exit_layer_first,
            sweep_arguments.exit_layer_last + 1,
            sweep_arguments.exit_layer_step,
        ):
            for num_speculations in range(
                sweep_arguments.num_speculations_first,
                sweep_arguments.num_speculations_last + 1,
                sweep_arguments.num_speculations_step,
            ):
                generation_config.exit_layer = exit_layer
                generation_config.num_speculations = num_speculations

                metric_result, _ = benchmark(
                    model, tokenizer, benchmark_arguments, generation_config, args.seed
                )

                results.append(
                    {
                        "exit_layer": exit_layer,
                        "num_speculations": num_speculations,
                        "acceptance_rate": metric_result["acceptance_rate"]["mean"],
                        "total_time": metric_result["total_time"]["mean"],
                        "time_per_token": metric_result["time_per_token"]["mean"],
                        "tokens_per_second": metric_result["tokens_per_second"]["mean"],
                    }
                )
                df = pd.DataFrame(results)
                # Update table every iteration
                df.to_csv(csv_fname, index=False)
                print(
                    f"exit_layer: {exit_layer}, num_speculations: {num_speculations}, time_per_token: {metric_result['time_per_token']['mean']}"
                )
    elif generation_config.generation_strategy in (
        "dynamic_early_exit_first",
        "dynamic_early_exit_max",
    ):
        for min_layer in range(
            sweep_arguments.min_layer_first,
            sweep_arguments.min_layer_last + 1,
            sweep_arguments.min_layer_step,
        ):
            for threshold in np.arange(
                sweep_arguments.threshold_first,
                sweep_arguments.threshold_last + sweep_arguments.threshold_step,
                sweep_arguments.threshold_step,
            ):
                for num_speculations in range(
                    sweep_arguments.num_speculations_first,
                    sweep_arguments.num_speculations_last + 1,
                    sweep_arguments.num_speculations_step,
                ):
                    generation_config.min_layer = min_layer
                    generation_config.threshold = threshold
                    generation_config.num_speculations = num_speculations

                    metric_result = benchmark(
                        model,
                        tokenizer,
                        benchmark_arguments,
                        generation_config,
                        args.seed,
                    )

                    results.append(
                        {
                            "min_layer": min_layer,
                            "num_speculations": num_speculations,
                            "threshold": threshold,
                            "acceptance_rate": metric_result["acceptance_rate"]["mean"],
                            "total_time": metric_result["total_time"]["mean"],
                            "time_per_token": metric_result["time_per_token"]["mean"],
                            "tokens_per_second": metric_result["tokens_per_second"][
                                "mean"
                            ],
                            "average_exit_layer": metric_result["avg_exit_layer"][
                                "mean"
                            ],
                        }
                    )
                    df = pd.DataFrame(results)
                    # Update table every iteration
                    df.to_csv(csv_fname, index=False)
                    print(
                        f"min_layer: {min_layer}, threshold: {threshold}, num_speculations: {num_speculations}, time_per_token: {metric_result['time_per_token']['mean']}"
                    )

    # Print summary table
    print("\n")
    header = results[0].keys()
    rows = [x.values() for x in results]
    print(tabulate.tabulate(rows, header))

    # Plot contour plot
    plot_contour(df, generation_config.generation_strategy, pdf_fname)


def plot_contour(df, generation_strategy: str, pdf_fname):
    ## Prepare grid coordinates (assuming exit_layer and num_speculations are integer indices)
    if generation_strategy == "self_speculative":
        x_column = "exit_layer"
    elif generation_strategy in ("dynamic_early_exit_max", "dynamic_early_exit_first"):
        x_column = "average_exit_layer"
    grid_x, grid_y = np.mgrid[
        df[x_column].min() : df[x_column].max() : 100j,
        df["num_speculations"].min() : df["num_speculations"].max() : 100j,
    ]
    ## Interpolate missing data
    grid_z = griddata(
        (df[x_column], df["num_speculations"]),
        df["tokens_per_second"],
        (grid_x, grid_y),
        method="linear",
    )
    ## Create the contour plot
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap="viridis")
    plt.colorbar(contour)
    ## Overlay the data points
    plt.scatter(df[x_column], df["num_speculations"], color="black", s=25, zorder=5)
    plt.title("Tokens Per Second")
    plt.xlabel(x_column)
    plt.ylabel("Number of Speculations")
    ## Save the plot
    plt.savefig(pdf_fname, format="pdf", dpi=300)
    ## Show the plot
    plt.show()


def process_cli_arguments() -> (
    Tuple[arguments.Arguments, BenchmarkArguments, GenerationConfig, SweepArguments]
):
    parser = transformers.HfArgumentParser(
        (arguments.Arguments, BenchmarkArguments, GenerationConfig, SweepArguments)
    )
    (
        general_arguments,
        benchmark_arguments,
        generation_config,
        sweep_arguments,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=False)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(
            general_arguments.model_args
        )
    else:
        general_arguments.model_args = {}

    return general_arguments, benchmark_arguments, generation_config, sweep_arguments


if __name__ == "__main__":
    args, benchmark_arguments, generation_config, sweep_arguments = (
        process_cli_arguments()
    )
    sweep(args, benchmark_arguments, generation_config, sweep_arguments)
