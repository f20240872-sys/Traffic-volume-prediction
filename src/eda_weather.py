"""Training-only weather and distribution plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COL
from src.split import chronological_split, load_processed_dataset


sns.set_theme(style="whitegrid", context="talk")


def training_observations(df: pd.DataFrame) -> pd.DataFrame:
    train, _, _ = chronological_split(df)
    return train.dropna(subset=[TARGET_COL]).copy()


def plot_weather_conditions(train: pd.DataFrame) -> None:
    order = train.groupby("weather_main")[TARGET_COL].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=train,
        x="weather_main",
        y=TARGET_COL,
        order=order,
        showfliers=False,
        color="#6baed6",
        ax=ax,
    )
    ax.set_title("Traffic volume by weather condition", fontweight="bold")
    ax.set_xlabel("Weather condition")
    ax.set_ylabel("Traffic volume (vehicles/hour)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "03_weather_conditions.png", dpi=240)
    plt.close(fig)


def plot_volume_distribution(train: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(train[TARGET_COL], bins=45, color="#2c7fb8", ax=ax)
    ax.set_title("Training-period traffic volume distribution", fontweight="bold")
    ax.set_xlabel("Traffic volume (vehicles/hour)")
    ax.set_ylabel("Observed hours")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_volume_distribution.png", dpi=240)
    plt.close(fig)


def plot_weather_correlations(train: pd.DataFrame) -> None:
    columns = ["temp", "rain_1h", "snow_1h", "clouds_all", TARGET_COL]
    corr = train[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Weather and traffic correlations", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_weather_correlations.png", dpi=240)
    plt.close(fig)


def plot_split_timeline(df: pd.DataFrame) -> None:
    train, val, test = chronological_split(df)
    spans = [
        ("Train", train.index.min(), train.index.max(), "#2c7fb8"),
        ("Validation", val.index.min(), val.index.max(), "#fdae61"),
        ("Test", test.index.min(), test.index.max(), "#d7191c"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    for label, start, end, color in spans:
        ax.barh([0], [(end - start).days], left=start, height=0.45, color=color, label=label)
        ax.text(start + (end - start) / 2, 0, label, ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Chronological model-development split", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.55), frameon=False)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_split_timeline.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_weather_eda() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_processed_dataset()
    train = training_observations(data)
    plot_weather_conditions(train)
    plot_volume_distribution(train)
    plot_weather_correlations(train)
    plot_split_timeline(data)
    print("Training-only weather EDA figures saved to reports/figures/.")


if __name__ == "__main__":
    run_weather_eda()
