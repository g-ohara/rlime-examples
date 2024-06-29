"""This module generates the experiment data for the paper."""

from __future__ import annotations

import csv
import multiprocessing
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from rlime.src.rlime import rlime_lime, utils
from rlime.src.rlime.rlime import HyperParam, explain_instance
from rlime.src.rlime.rlime_anchor import anchor
from rlime.src.rlime.sampler import Sampler
from rlime_examples.log import arg_to_log_level
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

if TYPE_CHECKING:
    from rlime.src.rlime.rlime_types import Classifier, Dataset, IntArray, Rule

logger = getLogger(__name__)


def sample_to_csv(tab: list[tuple[str, str]], path: str) -> None:
    """Save the sample as a CSV file."""
    with Path(path).open(mode="w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for feature, sample in tab:
            writer.writerow([feature, sample])


def save_weights(
    path: str,
    weights: list[float],
    rule_info: tuple[Rule, float, float] | None = None,
) -> None:
    """Save the weights as a CSV file.

    Parameters
    ----------
    weights : list[float]
        The weights to be saved.
    path : str
        The path to the CSV file.
    rule_info : tuple[Rule, float, float] | None
        The rule information to be saved.

    """
    with Path(path).open(mode="w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(weights)
        if rule_info is not None:
            rule, precision, coverage = rule_info
            writer.writerow(rule)
            writer.writerow([precision, coverage])


def main() -> None:
    """The main function of the module."""
    # Get log level.
    arg_to_log_level()

    # Load the dataset.
    dataset = utils.load_dataset(
        "recidivism", "src/rlime/src/rlime/datasets/", balance=True
    )

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    # Get the target instances.
    sample_num = 50
    idx_list = list(range(sample_num))
    trgs: list[IntArray] = []
    labels_trgs: list[IntArray] = []

    # Save the target instances as CSV files.
    for idx in idx_list:
        trg, label, tab = utils.get_trg_sample(idx, dataset)
        trgs.append(trg)
        labels_trgs.append(label)
        sample_to_csv(tab, f"examples/{idx:04d}.csv")

    predict: Classifier = black_box.predict
    args = [(idx, trg, dataset, predict) for idx, trg in zip(idx_list, trgs)]

    with multiprocessing.Pool() as pool:
        pool.starmap(generate_examples, tqdm(args))


def generate_examples(
    idx: int, trg: IntArray, dataset: Dataset, black_box: Classifier
) -> None:
    """Generate the LIME and R-LIME explanations for the given sample."""
    logger.info("Target instance: %s", idx)

    # Generate the LIME explanation and save it as an image.
    logger.info(" LIME")
    generate_lime(trg, dataset, black_box, f"examples/lime-{idx:04d}.csv")

    # Generate the Anchor explanation and save it as an image.
    logger.info(" Anchor")
    for tau in [70, 80, 90]:
        generate_anchor(
            trg,
            dataset,
            black_box,
            f"examples/anchor-{idx:04d}-{tau}.csv",
            tau,
        )

    # Generate the R-LIME explanation and save it as an image.
    logger.info(" R-LIME")
    hyper_param = HyperParam()
    for hyper_param.tau in [0.7, 0.8, 0.9]:
        logger.info("  tau = %s", hyper_param.tau)
        generate_rlime(
            trg,
            dataset,
            black_box,
            f"examples/newlime-{idx:04d}-{int(hyper_param.tau * 100)}.csv",
            hyper_param,
        )


def generate_lime(
    trg: IntArray, dataset: Dataset, black_box: Classifier, img_name: str
) -> None:
    """Generate the LIME explanation for the given sample."""
    # Generate the LIME explanation.
    sampler = Sampler(trg, dataset.train, black_box, dataset.categorical_names)
    coef, _ = rlime_lime.explain(trg, sampler, 100000)

    # Save the LIME explanation as an image.
    save_weights(img_name, coef)


def generate_anchor(
    trg: IntArray,
    dataset: Dataset,
    black_box: Classifier,
    img_name: str,
    tau: float,
) -> None:
    """Generate the Anchor explanation for the given sample."""
    try:
        rule_strs, acc, cov = anchor(trg, dataset, black_box, tau / 100)
        with Path(img_name).open(mode="w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rule_strs)
            writer.writerow([acc, cov])

    except Exception as e:
        logger.warning("  No explanation found.")
        logger.warning("  %s", e)


def generate_rlime(
    trg: IntArray,
    dataset: Dataset,
    black_box: Classifier,
    img_name: str,
    hyper_param: HyperParam,
) -> None:
    """Generate the R-LIME explanations for the given sample."""
    # Generate the R-LIME explanation and standardize its weights.
    result = explain_instance(trg, dataset, black_box, hyper_param)
    if result is None:
        logger.warning("   No explanation found. (img_name: %s)", img_name)
        return
    _, arm = result

    weights: list[float] = list(
        arm.surrogate_model["LogisticRegression"].weights.values()
    )
    weights = [w / sum(map(abs, weights)) for w in weights]

    # Save the R-LIME explanation as an image.
    rule_info = (arm.rule, arm.n_rewards / arm.n_samples, arm.coverage)
    save_weights(img_name, weights, rule_info)


if __name__ == "__main__":
    main()
