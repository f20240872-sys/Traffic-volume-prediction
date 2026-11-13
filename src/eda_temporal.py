import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COL
from src.split import load_processed_dataset

sns.set_theme(style="whitegrid", context="talk")


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract helper calendar fields from the DatetimeIndex for EDA plotting."""
    eda_df = df.dropna(subset=[TARGET_COL]).copy()
    eda_df["hour"] = eda_df.index.hour
    eda_df["dayofweek"] = eda_df.index.dayofweek  # 0=Mon, 6=Sun
    eda_df["day_name"] = eda_df.index.day_name()
    eda_df["is_weekend"] = np.where(eda_df["dayofweek"] >= 5, "Weekend (Sat-Sun)", "Weekday (Mon-Fri)")
    eda_df["month"] = eda_df.index.month
    eda_df["month_name"] = eda_df.index.strftime("%b")

    if "holiday" in eda_df.columns:
        eda_df["is_holiday"] = np.where(
            eda_df["holiday"].fillna("None") != "None", "Holiday", "Non-Holiday"
        )
    else:
        eda_df["is_holiday"] = "Non-Holiday"

    return eda_df


def plot_hourly_weekday_vs_weekend(df: pd.DataFrame) -> None:
    """Plot 1: Hourly traffic profile comparing Weekdays vs Weekends (mean + IQR)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    palette = {"Weekday (Mon-Fri)": "#1f77b4", "Weekend (Sat-Sun)": "#ff7f0e"}

    for group, color in palette.items():
        sub = df[df["is_weekend"] == group]
        stats = sub.groupby("hour")[TARGET_COL].agg(
            mean="mean",
            q25=lambda s: np.percentile(s, 25),
            q75=lambda s: np.percentile(s, 75),
        )
        ax.plot(stats.index, stats["mean"], label=f"{group} (Mean)", color=color, linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(stats.index, stats["q25"], stats["q75"], color=color, alpha=0.18, label=f"{group} (25th–75th pct)")

    ax.set_title("Hourly Traffic Volume Profile: Weekday Rush Hours vs. Weekend", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day (0 = Midnight, 17 = 5 PM)")
    ax.set_ylabel("Traffic Volume (vehicles/hour)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out = FIGURES_DIR / "01_hourly_weekday_vs_weekend.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out}")


def plot_dow_hour_heatmap(df: pd.DataFrame) -> None:
    """Plot 2: Day-of-Week x Hour-of-Day average traffic volume heatmap."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.pivot_table(index="day_name", columns="hour", values=TARGET_COL, aggfunc="mean").reindex(day_order)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=False,
        cbar_kws={"label": "Mean Traffic Volume (veh/hr)"},
        ax=ax,
    )
    ax.set_title("Average Hourly Traffic Volume by Day of Week", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    plt.tight_layout()

    out = FIGURES_DIR / "02_dow_hour_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out}")


def plot_monthly_seasonality(df: pd.DataFrame) -> None:
    """Plot 3: Monthly seasonality (mean line + boxplot distribution)."""
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    present_months = [m for m in month_order if m in df["month_name"].unique()]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(
        data=df,
        x="month_name",
        y=TARGET_COL,
        order=present_months,
        color="#8ecae6",
        fliersize=1.5,
        ax=ax,
    )
    monthly_means = df.groupby("month_name")[TARGET_COL].mean().reindex(present_months)
    ax.plot(range(len(present_months)), monthly_means.values, color="#d62828", marker="o", linewidth=2, label="Monthly Mean")

    ax.set_title("Monthly Seasonality in Interstate Traffic Volume", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Traffic Volume (vehicles/hour)")
    ax.legend(frameon=True)
    plt.tight_layout()

    out = FIGURES_DIR / "03_monthly_seasonality.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out}")


def plot_holiday_effect(df: pd.DataFrame) -> None:
    """Plot 4: Hourly traffic volume on Holidays vs Non-Holidays."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.lineplot(
        data=df,
        x="hour",
        y=TARGET_COL,
        hue="is_holiday",
        style="is_holiday",
        markers=True,
        dashes=False,
        errorbar=("ci", 95),
        palette={"Non-Holiday": "#2b2d42", "Holiday": "#ef233c"},
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Holiday Effect: Hourly Traffic Volume (Holiday vs. Regular Day)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Traffic Volume (vehicles/hour)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(title="Day Type", frameon=True)
    plt.tight_layout()

    out = FIGURES_DIR / "04_holiday_effect.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[Saved] {out}")


def run_temporal_eda() -> None:
    """Generate all 4 temporal EDA figures and save to reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed_dataset()
    eda_df = add_temporal_columns(df)

    plot_hourly_weekday_vs_weekend(eda_df)
    plot_dow_hour_heatmap(eda_df)
    plot_monthly_seasonality(eda_df)
    plot_holiday_effect(eda_df)
    print("\n[Done] All temporal EDA figures saved to reports/figures/.")


if __name__ == "__main__":
    run_temporal_eda()