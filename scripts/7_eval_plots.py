import modal

app = modal.App("rqae-upload-gemmascope-features")
volume = modal.Volume.from_name("rqae-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch==2.2.1", gpu="t4")
    .pip_install("scipy==1.11.4")
    .pip_install("modal==0.64.10")
    .pip_install("huggingface-hub==0.24.5")
    .pip_install("safetensors==0.4.2")
    .pip_install("transformers==4.44.0")
    .pip_install("tqdm")
    .pip_install("anthropic")
)


with image.imports():
    from rqae.model import RQAE
    from rqae.llm import Gemma2
    from rqae.gemmascope import JumpReLUSAE
    from rqae.feature import Feature
    from transformers import AutoTokenizer
    from tqdm import tqdm
    import numpy as np


def plot_model_metrics(
    data,
    model_names=None,
    metric_names=None,
    model_colors=[],
    title=None,
    show=[],
    category_lengths=[],
    figsize=(12, 6),
):
    """
    Create a grouped bar chart comparing metrics across models.

    Args:
        data: numpy array of shape (B, D, 2) where:
            B = number of models
            D = number of metrics
            Last dim contains (mean, variance) pairs
        model_names: list of B model names (optional)
        metric_names: list of D metric names (optional)
        title: plot title (optional)
        figsize: tuple specifying figure dimensions (optional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if model_names is None:
        model_names = [f"Model {i}" for i in range(data.shape[0])]
    if metric_names is None:
        metric_names = [f"Metric {i}" for i in range(data.shape[1])]

    num_models = len(model_names)
    num_metrics = len(metric_names)
    category_lengths = np.cumsum(category_lengths)

    # Set up the plot
    plt.figure(figsize=figsize)

    # Calculate bar positions
    bar_width = 0.8 / num_models
    metric_positions = np.arange(num_metrics)

    # Plot bars for each model
    plotted_models = 0
    for i in range(num_models):
        num_category_jumps = sum(
            [1 if i >= category_lengths[j] else 0 for j in range(len(category_lengths))]
        )
        positions = (
            metric_positions
            + (plotted_models - num_models / 2 + 0.5) * bar_width
            + num_category_jumps * 0.04
        )
        means = data[i, :num_metrics, 0]
        errors = data[i, :num_metrics, 1]  # Standard deviation

        means_mask = means > 0
        means = means[means_mask]
        errors = errors[means_mask]
        positions = positions[means_mask]
        if show[i]:
            plt.bar(
                positions,
                means,
                bar_width,
                label=model_names[i],
                yerr=errors,
                capsize=5,
                color=model_colors[i],
            )
            plotted_models += 1

    # Customize the plot
    # plt.xlabel("Metrics")
    plt.ylabel("Score")
    # if title:
    #     plt.title(title)

    plt.xticks(metric_positions - 0.15, metric_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return plt.gcf()


@app.function(volumes={"/data": volume}, image=image, timeout=60 * 60)
def get_metrics_for_models(
    metrics=["detection"],
    models=[
        "rqae-rqae-round_fsq-cbd4-cbs5-nq1024|16",
        "gemmascope-gemma-2-2b-res-12-w16k-l82",
    ],
    dataset="monology_pile",
    num_features=100,
):
    from rqae.feature import RQAEFeature, Feature
    from tqdm import tqdm
    import numpy as np
    import os

    results = np.ones((len(models), len(metrics), num_features)) * -1

    for m, model in enumerate(tqdm(models, desc="Models")):
        if "rqae" in model:
            if "|" in model:
                layer = int(model.split("|")[1])
                model = model.split("|")[0]
            else:
                layer = -1
        all_features = os.listdir(f"/data/datasets/{dataset}/features/{model}")
        all_features = sorted(all_features, key=lambda x: int(x.split(".")[0]))
        for f, feature in tqdm(enumerate(all_features), desc="Features"):
            if "rqae" in model:
                feature = RQAEFeature.load(
                    f"/data/datasets/{dataset}/features/{model}/{feature}"
                )
                feature = feature.to_feature(layer=feature.layers.tolist().index(layer))
            else:
                feature = Feature.load(
                    f"/data/datasets/{dataset}/features/{model}/{feature}"
                )

            for i, metric in enumerate(metrics):
                if metric in feature.scores:
                    results[m, i, f] = feature.scores[metric]
    return results


model_names = [
    "RQAE (16 layers)",
    "RQAE (64 layers)",
    "RQAE (256 layers)",
    "RQAE (1023 layers)",
    "16k L0=22",
    "16k L0=41",
    "16k L0=82",
    "16k L0=176",
    "16k L0=445",
    "32k L0=76",
    "65k L0=72",
    "262k L0=67",
    "524k L0=65",
]
metric_names = ["Detection", "Fuzzing"]
metric_ids = ["detection", "fuzzing"]
model_ids = [
    "rqae-rqae-round_fsq-cbd4-cbs5-nq1024|16",
    "rqae-rqae-round_fsq-cbd4-cbs5-nq1024|64",
    "rqae-rqae-round_fsq-cbd4-cbs5-nq1024|256",
    "rqae-rqae-round_fsq-cbd4-cbs5-nq1024|1023",
    "gemmascope-gemma-2-2b-res-12-w16k-l22",
    "gemmascope-gemma-2-2b-res-12-w16k-l41",
    "gemmascope-gemma-2-2b-res-12-w16k-l82",
    "gemmascope-gemma-2-2b-res-12-w16k-l176",
    "gemmascope-gemma-2-2b-res-12-w16k-l445",
    "gemmascope-gemma-2-2b-res-12-w32k-l76",
    "gemmascope-gemma-2-2b-res-12-w65k-l72",
    "gemmascope-gemma-2-2b-res-12-w262k-l67",
    "gemmascope-gemma-2-2b-res-12-w524k-l65",
]
show = [
    # rqae
    True,
    True,
    True,
    False,
    # l0
    False,
    False,
    True,
    False,
    False,
    # width
    True,
    True,
    True,
    True,
]
category_lengths = [4, 9]


@app.local_entrypoint()
def main():

    results = get_metrics_for_models.remote(
        metrics=metric_ids,
        models=model_ids,
        num_features=100,
    )
    import numpy as np

    np.save("results.npy", results)


if __name__ == "__main__":
    from scripts.plotting_utils import *

    results = np.load("results.npy")

    # plt.hist(results[0, 0, :])
    # plt.show()

    mean = results.mean(2)
    std = results.std(2)
    # Convert to standard error
    std = std / np.sqrt(results.shape[2])

    cm = create_blog_colormaps()["accent_gradient"]
    cm2 = create_blog_colormaps()["primary_accent"]
    cm3 = create_blog_colormaps()["full_palette"]

    rqae_model_colors = [cm((i + 1) / 4) for i in range(4)]
    l0_model_colors = [cm2((i + 2) / 10) for i in range(5)][::-1]
    width_model_colors = [cm2((i + 2) / 10) for i in range(4)][::-1]

    model_colors = rqae_model_colors + l0_model_colors + width_model_colors

    fig = plot_model_metrics(
        np.stack([mean, std], 2),
        model_names=model_names,
        metric_names=metric_names,
        model_colors=model_colors,
        show=show,
        category_lengths=category_lengths,
        title="Model Performance Comparison",
    )
    # plt.show()
    plt.savefig("eval.jpg", dpi=300, bbox_inches="tight")
