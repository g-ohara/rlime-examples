"""This module generates the experiment data for the paper."""

import csv
import multiprocessing

from rlime.src.rlime import rlime_lime, utils
from rlime.src.rlime.rlime import HyperParam, explain_instance
from rlime.src.rlime.rlime_types import Classifier, Dataset, IntArray, Rule
from rlime.src.rlime.sampler import Sampler
from sklearn.ensemble import RandomForestClassifier  # type: ignore


def sample_to_csv(tab: list[tuple[str, str]], path: str) -> None:
    """Save the sample as a CSV file."""
    with open(path, "w", encoding="utf-8") as f:
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
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(weights)
        if rule_info is not None:
            rule, precision, coverage = rule_info
            writer.writerow(rule)
            writer.writerow([precision, coverage])


def main() -> None:
    """The main function of the module."""
    # Load the dataset.
    dataset = utils.load_dataset(
        "recidivism", "src/rlime/src/rlime/datasets/", balance=True,
    )

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    # Get the target instances.
    sample_num = 50
    idx_list = list(range(sample_num))
    trgs = []
    labels_trgs = []

    # Save the target instances as CSV files.
    for idx in idx_list:
        trg, label, tab = utils.get_trg_sample(idx, dataset)
        trgs.append(trg)
        labels_trgs.append(label)
        sample_to_csv(tab, f"examples/{idx:04d}.csv")

    with multiprocessing.Pool() as pool:
        pool.starmap(
            generate_lime_and_rlime,
            [
                (idx, trg, dataset, black_box.predict)
                for idx, trg in zip(idx_list, trgs)
            ],
        )


def generate_lime_and_rlime(
    idx: int,
    trg: IntArray,
    dataset: Dataset,
    black_box: RandomForestClassifier,
) -> None:
    """Generate the LIME and R-LIME explanations for the given sample."""
    print(f"Target instance: {idx}")

    # Generate the LIME explanation and save it as an image.
    print("LIME")
    generate_lime(trg, dataset, black_box, f"examples/lime-{idx:04d}.csv")

    # Generate the R-LIME explanation and save it as an image.
    print("R-LIME")
    hyper_param = HyperParam()
    for hyper_param.tau in [0.7, 0.8, 0.9]:
        print(f"tau = {hyper_param.tau}")
        generate_rlime(
            trg,
            dataset,
            black_box,
            f"examples/newlime-{idx:04d}-{int(hyper_param.tau * 100)}.csv",
            hyper_param,
        )


def generate_lime(
    trg: IntArray, dataset: Dataset, black_box: Classifier, img_name: str,
) -> None:
    """Generate the LIME explanation for the given sample."""
    # Generate the LIME explanation.
    sampler = Sampler(trg, dataset.train, black_box, dataset.categorical_names)
    coef, _ = rlime_lime.explain(trg, sampler, 100000)

    # Save the LIME explanation as an image.
    save_weights(img_name, coef)


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
        print("No explanation found.")
        return
    names, arm = result
    weights = list(arm.surrogate_model["LogisticRegression"].weights.values())
    weights = [w / sum(map(abs, weights)) for w in weights]

    # Save the R-LIME explanation as an image.
    rule_info = (arm.rule, arm.n_rewards / arm.n_samples, arm.coverage)
    save_weights(img_name, weights, rule_info)


if __name__ == "__main__":
    main()
