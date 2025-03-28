import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ranks(results: pd.DataFrame):
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    num_models = len(results.index)
    palette = sns.color_palette("husl", num_models)

    for i, model in enumerate(results.index):
        plt.plot(
            results.columns,
            results.loc[model],
            marker="o",
            markersize=8,
            linewidth=2,
            label=model,
            color=palette[i],
        )

    plt.title("Model Rankings Across Tasks", fontsize=16)
    plt.xlabel("Tasks", fontsize=14)
    plt.ylabel("Rank (lower is better)", fontsize=14)

    plt.ylim(results.values.max() + 0.5, 0.5)
    plt.yticks(range(1, int(results.values.max()) + 1))

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)

    plt.tight_layout()

    plt.grid(True)
    plt.show()


def plot_corr_matrix(results: pd.DataFrame):
    task_spearman_corr = results.corr(method="spearman")
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    mask = np.triu(np.ones_like(task_spearman_corr), k=0).T
    sns.heatmap(
        task_spearman_corr[:-1],
        mask=mask[:-1],
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes,
        fmt=".2f",
    )
    axes.set_title("Spearman Correlation Between Tasks", fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_radar_chart(results: pd.DataFrame):
    labels = results.columns  # Task names (axes)
    num_axes = len(labels)

    # Radar plots require all axes to form a closed shape
    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False).tolist()
    # Close the circle by repeating the first angle at the end
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for model in results.index:
        values = results.loc[model].tolist()
        values += values[:1]  # Close the circle
        ax.plot(angles, values, label=model)
        # ax.fill(angles, values, alpha=0.25)  # Fill the area under the curve

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title("Task Performances", size=15, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.show()
