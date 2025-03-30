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
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
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
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha="right")
    axes.set_yticklabels(axes.get_yticklabels(), rotation=45)


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

def ranks_chart(rank_stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the horizontal bars for mean ranks
    bars = ax.barh(rank_stats.index, rank_stats['Mean Rank'], height=0.5, color='skyblue')

    # Add error bars representing standard deviation
    ax.errorbar(rank_stats['Mean Rank'], rank_stats.index, xerr=rank_stats['Std Dev'],
                fmt='none', ecolor='black', capsize=5, elinewidth=2, markeredgewidth=2)

    # Customize the plot
    ax.set_xlabel('Mean Rank', fontsize=12)
    # ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Model Performance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # To have the best model at the top
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add text annotations for mean rank and std dev, rounding std dev to 1 decimal place
    for i, (mean, std) in enumerate(zip(rank_stats['Mean Rank'], rank_stats['Std Dev'])):
        # Round std to 1 decimal place
        rounded_std = round(std, 1)
        ax.text(mean + rounded_std + 0.4, i, f"{mean} ± {rounded_std}", va='center',
                fontsize=10)  # Increased horizontal offset

    # Set x-axis to start from 0 and extend to cover all error bars
    max_value = max(rank_stats['Mean Rank'] + rank_stats['Std Dev']) + 1
    ax.set_xlim(0, max_value + 1)

    # Add a note about interpretation
    # plt.figtext(0.5, 0.01, "Note: Lower rank is better. Error bars represent standard deviation.",
    #             ha="center", fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()