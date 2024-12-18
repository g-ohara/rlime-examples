"""Module for plotting images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from rlime.src.rlime.utils import get_trg_sample, load_dataset
from rlime_examples.log import arg_to_log_level
from tqdm import tqdm

if TYPE_CHECKING:
    from rlime.src.rlime.rlime_types import IntArray, Rule

logger = getLogger(__name__)


@dataclass
class RuleInfo:
    """Rule information."""

    rule_str: list[str]
    precision: float
    coverage: float


def main() -> None:
    """The main function of the module."""
    # Get log level.
    arg_to_log_level()

    # Load the dataset.
    dataset = load_dataset(
        "recidivism", "src/rlime/src/rlime/datasets/", balance=True
    )

    for idx in tqdm(range(50)):
        logger.info("Target instance: %s", idx)
        logger.info(" LIME:")
        trg, _, _ = get_trg_sample(idx, dataset)

        # Load the weights.
        csv_name = f"examples/lime-{idx:04d}.csv"
        result = load_weights(
            csv_name,
            trg,
            dataset.feature_names,
            dataset.categorical_names,
            dataset.ordinal_features,
        )

        # Plot the weights.
        if result is None:
            logger.warning("  Not found.")
        else:
            logger.debug("  Found.")
            weights, _ = result
            img_name = f"examples/lime-{idx:04d}.eps"
            plot_weights(weights, dataset.feature_names, img_name=img_name)

        logger.info(" Anchor:")
        for tau in [70, 80, 90]:
            csv_name = f"examples/anchor-{idx:04d}-{tau}.csv"
            img_name = f"examples/anchor-{idx:04d}-{tau}.eps"
            plot_anchor(csv_name, img_name)

        logger.info(" R-LIME:")
        for tau in [70, 80, 90]:
            # Load the weights.
            logger.debug("  tau = %s:", tau)
            csv_name = f"examples/newlime-{idx:04d}-{tau}.csv"
            result = load_weights(
                csv_name,
                trg,
                dataset.feature_names,
                dataset.categorical_names,
                dataset.ordinal_features,
            )

            # Plot the weights.
            if result is None:
                logger.warning("  Not found.")
            else:
                logger.debug("  Found.")
                weights, rule_info = result
                img_name = f"examples/newlime-{idx:04d}-{tau}.eps"
                plot_weights(
                    weights,
                    dataset.feature_names,
                    rule_info,
                    img_name=img_name,
                )


def load_weights(
    path: str,
    trg: IntArray,
    feature_names: list[str],
    categorical_names: dict[int, list[str]],
    ordinal_features: list[int],
) -> tuple[list[float], RuleInfo | None] | None:
    """Load the weights from a CSV file.

    Parameters
    ----------
    path : str
        The path to the CSV file.

    Returns:
    -------
    tuple[list[float], RuleInfo | None]
        The weights and the rule information.

    """

    def get_names(
        trg: IntArray,
        rule: Rule,
        feature_names: list[str],
        categorical_names: dict[int, list[str]],
        ordinal_features: list[int],
    ) -> list[str]:
        """Get the names of the features in the rule."""
        names: list[str] = []
        for r in rule:
            name = categorical_names[r][int(trg[r])]
            if r not in ordinal_features:
                name = feature_names[r] + " = " + name
            names.append(name)
        return names

    try:
        with Path(path).open(encoding="utf-8") as f:
            reader = csv.reader(f)
            weights = list(map(float, next(reader)))
            rule_info = None
            try:
                rule = tuple(map(int, next(reader)))
                rule_str = get_names(
                    trg,
                    rule,
                    feature_names,
                    categorical_names,
                    ordinal_features,
                )
                precision, coverage = next(reader)
                rule_info = RuleInfo(
                    rule_str, float(precision), float(coverage)
                )
            except StopIteration:
                pass
            return weights, rule_info
    except FileNotFoundError:
        logger.warning("File %s not found.", path)
        return None


def concat_names(names: list[str]) -> str:
    """Concatenate the names to multiline string.

    Concatenate the names to multiline string,
    such that the string length is less than the specified length.
    """
    multiline_names: list[str] = []
    line: list[str] = []
    line_len = 0
    and_len = len(" AND ")
    max_len = 50

    for name in names:
        if line_len + and_len + len(name) > max_len and len(line) > 0:
            multiline_names.append(" AND ".join(line))
            line = [name]
            line_len = len(name)
        else:
            line.append(name)
            line_len += len(name) + and_len

    multiline_names.append(" AND ".join(line))
    return " AND \n".join(multiline_names)


def plot_anchor(csv_name: str, img_name: str) -> None:
    """Create image of the output of Anchor.

    Parameters
    ----------
    csv_name : str
        The name of the CSV file from which the output is loaded.
    img_name : str
        The name of the image to be created.
    """
    # Load the CSV file.
    logger.debug("  Loading %s ...", csv_name)
    try:
        with Path(csv_name).open(encoding="utf-8") as f:
            reader = csv.reader(f)
            predicate_strs = list(map(str, next(reader)))
            acc_str, cov_str = next(reader)
    except FileNotFoundError:
        logger.warning("File %s not found.", csv_name)
        return
    rule_str = concat_names(predicate_strs)
    acc_percent = 100 * float(acc_str)
    cov_percent = 100 * float(cov_str)

    # Plot the image.
    logger.debug("  Plotting %s ...", img_name)
    fig, ax = plt.subplots()  # type: ignore
    ax.set_visible(False)
    fig.suptitle(  # type: ignore
        f"{rule_str}\n"
        f"with Accuracy {acc_percent:.2f}% "
        f"and Coverage {cov_percent:.2f}%",
        fontsize=15,
    )
    Path.mkdir(Path(img_name).parent, parents=True, exist_ok=True)
    plt.savefig(img_name, bbox_inches="tight")  # type: ignore
    plt.close()


def plot_weights(
    weights: list[float],
    feature_names: list[str],
    rule_info: RuleInfo | None = None,
    img_name: str | None = None,
) -> None:
    """Plot the weights of the surrogate model.

    Parameters
    ----------
    weights : list[float]
        The weights of the features
    feature_names : list[str]
        The names of the features
    rule_info : RuleInfo, optional
        The rule string, accuracy and coverage, by default None
    img_name : str, optional
        The name of the image, by default None

    Returns:
    -------
    None

    """
    features = feature_names
    abs_values = [abs(x) for x in weights]
    _, sorted_features, sorted_values = zip(
        *sorted(zip(abs_values, features, weights), reverse=False)[-5:]
    )
    plt.figure()  # type: ignore
    color = [
        "#32a852" if sorted_values[i] > 0 else "#cf4529"
        for i in range(len(sorted_values))
    ]
    plt.rc("ytick", labelsize=12)  # type: ignore
    plt.barh(sorted_features, sorted_values, color=color)  # type: ignore

    if rule_info is not None:
        anchor_str = concat_names(rule_info.rule_str)
        acc_perc = rule_info.precision * 100
        cov_perc = rule_info.coverage * 100
        plt.title(  # type: ignore
            f"{anchor_str}\n"
            f"with Accuracy {acc_perc:.2f}% "
            f"and Coverage {cov_perc:.2f}%",
            fontsize=15,
        )

    for f, v in zip(sorted_features, sorted_values):
        plt.text(v, f, round(v, 5), fontsize=12)  # type: ignore

    if img_name is not None:
        Path.mkdir(Path(img_name).parent, parents=True, exist_ok=True)
        plt.savefig(img_name, bbox_inches="tight")  # type: ignore

    plt.close()


if __name__ == "__main__":
    main()
