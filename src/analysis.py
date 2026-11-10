"""End-to-end data pipeline used by the notebook and command-line interface."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .flight_risk import find_flight_risks
from .modeling import evaluate_forecast, make_forecast_frame
from .scoring import monthly_rankings, monthly_scores
from .sentiment import classify_messages, inspect_dataframe, resolve_column

logger = logging.getLogger(__name__)
SCORE_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def pseudonymize(values: pd.Series) -> tuple[pd.Series, dict[str, str]]:
    """Map identities to stable project-local IDs without exporting source identities."""
    unique = sorted(values.dropna().astype(str).unique())
    mapping = {value: f"employee_{i:04d}" for i, value in enumerate(unique, start=1)}
    return values.astype("string").map(mapping), mapping


def _save_charts(messages: pd.DataFrame, monthly: pd.DataFrame, sentiment_col: str,
                 visualizations_dir: Path, employee_col: str = "employee") -> None:
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    counts = messages[sentiment_col].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False,
                palette=["#4c9f70", "#9aa0a6", "#d95f59"], ax=ax)
    ax.set(title="Message sentiment distribution", xlabel="Predicted sentiment", ylabel="Messages")
    fig.tight_layout(); fig.savefig(visualizations_dir / "sentiment_distribution.png", dpi=160); plt.close(fig)

    activity = messages.dropna(subset=["_analysis_month"]).groupby("_analysis_month").size()
    fig, ax = plt.subplots(figsize=(10, 4)); activity.plot(ax=ax, marker="o")
    ax.set(title="Message activity by calendar month", xlabel="Month", ylabel="Messages")
    fig.tight_layout(); fig.savefig(visualizations_dir / "monthly_activity.png", dpi=160); plt.close(fig)

    valid = messages.dropna(subset=["_analysis_month", sentiment_col]).copy()
    trend = valid.groupby(["_analysis_month", sentiment_col]).size().unstack(fill_value=0)
    trend = trend.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4)); trend.plot(ax=ax, marker="o")
    ax.set(title="Monthly predicted sentiment counts", xlabel="Month", ylabel="Messages")
    fig.tight_layout(); fig.savefig(visualizations_dir / "monthly_sentiment_trend.png", dpi=160); plt.close(fig)

    if employee_col not in monthly.columns:
        raise ValueError(f"Monthly score table is missing its employee key column {employee_col!r}")
    heatmap = monthly.pivot(index=employee_col, columns="month", values="monthly_score")
    if not heatmap.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(heatmap.columns) * .7), max(4, min(14, len(heatmap) * .25))))
        sns.heatmap(heatmap, cmap="vlag", center=0, ax=ax)
        ax.set(title="Monthly employee sentiment score (pseudonymous)", xlabel="Month", ylabel="Employee ID")
        fig.tight_layout(); fig.savefig(visualizations_dir / "monthly_sentiment_heatmap.png", dpi=160); plt.close(fig)


def run_analysis(data_path: str | Path, output_dir: str | Path = "outputs",
                 visualizations_dir: str | Path = "visualizations", text_col: str | None = None,
                 employee_col: str | None = None, date_col: str | None = None,
                 model_name: str = MODEL_NAME, batch_size: int = 16) -> dict:
    """Run the project pipeline; all tabular artifacts are written under output_dir."""
    data_path = Path(data_path)
    if data_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv dataset; refusing to parse {data_path.name!r} as CSV.")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    output_dir, visualizations_dir = Path(output_dir), Path(visualizations_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw = pd.read_csv(data_path, low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Could not read {data_path} as a valid CSV: {exc}") from exc
    if raw.empty:
        raise ValueError("Dataset has a header but no records.")
    profile = inspect_dataframe(raw)
    (output_dir / "schema_profile.json").write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")

    text_col = resolve_column(raw, text_col, "text")
    employee_col = resolve_column(raw, employee_col, "employee")
    date_col = resolve_column(raw, date_col, "date")
    if len({text_col, employee_col, date_col}) != 3:
        raise ValueError("Text, employee, and date must be three distinct dataset columns.")
    logger.info("Dataset shape=%s; selected text=%s employee=%s date=%s", raw.shape, text_col, employee_col, date_col)

    frame = raw.copy()
    frame["_source_row"] = np.arange(len(frame), dtype=int)
    frame["_parsed_date"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    frame["sentiment"] = classify_messages(frame[text_col], model_name=model_name, batch_size=batch_size)
    frame["employee_id"], identity_map = pseudonymize(frame[employee_col])
    frame["_analysis_month"] = frame["_parsed_date"].dt.strftime("%Y-%m")
    frame["word_count"] = frame[text_col].fillna("").astype(str).str.split().str.len()
    frame["message_length"] = frame[text_col].fillna("").astype(str).str.len()

    valid = frame.dropna(subset=["employee_id", "_parsed_date"]).copy()
    logger.info("Usable time-series rows=%d; invalid dates=%d; missing identities=%d; blank messages=%d",
                len(valid), int(frame["_parsed_date"].isna().sum()), int(frame[employee_col].isna().sum()),
                int(frame[text_col].isna().sum() + frame[text_col].astype("string").str.strip().eq("").sum()))
    score_input = valid.rename(columns={"employee_id": "employee", "_parsed_date": "analysis_date"})
    message_scores, monthly = monthly_scores(score_input, "employee", "analysis_date")
    positive, negative = monthly_rankings(monthly, "employee")
    risks = find_flight_risks(score_input.assign(sentiment=valid["sentiment"].values), "employee", "analysis_date")

    scored_messages = message_scores.rename(columns={"employee": "employee_id", "analysis_date": "date"})
    expected_scores = scored_messages.groupby("month")["sentiment_score"].sum().sort_index()
    observed_scores = monthly.groupby("month")["monthly_score"].sum().sort_index()
    if not expected_scores.equals(observed_scores):
        raise AssertionError("Monthly score totals do not reconcile to message-level scores.")
    for table in (positive, negative):
        if not table.empty and (table.groupby("month").size() > 3).any():
            raise AssertionError("Monthly ranking includes more than three employees.")
    if not risks.empty:
        durations = risks["window_end"] - risks["window_start"]
        if (durations > pd.Timedelta(days=30)).any() or (risks["negative_email_count"] < 4).any():
            raise AssertionError("A reported flight-risk interval fails the four-negative/30-day rule.")
    scored_messages.to_csv(output_dir / "message_scores.csv", index=False)
    monthly.rename(columns={"employee": "employee_id"}).to_csv(output_dir / "monthly_employee_scores.csv", index=False)
    positive.rename(columns={"employee": "employee_id"}).to_csv(output_dir / "monthly_top_positive.csv", index=False)
    negative.rename(columns={"employee": "employee_id"}).to_csv(output_dir / "monthly_top_negative.csv", index=False)
    risks.rename(columns={"employee": "employee_id"}).to_csv(output_dir / "flight_risk_report.csv", index=False)

    # Labeled export contains only the selected message, date, sentiment, and a pseudonym.
    # Keep outputs/ access-controlled: message text itself may contain confidential content.
    labeled = pd.DataFrame({"record_id": frame["_source_row"], "employee_id": frame["employee_id"],
                            "date": frame["_parsed_date"], "message": frame[text_col], "sentiment": frame["sentiment"]})
    labeled.to_csv(output_dir / "labeled_messages.csv", index=False)

    counts = frame["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    activity = (monthly[["employee", "month", "message_count"]]
                .rename(columns={"employee": "employee_id"}))
    q1, q3 = activity["message_count"].quantile([.25, .75]) if not activity.empty else (0, 0)
    upper = float(q3 + 1.5 * (q3 - q1))
    anomalies = activity[activity["message_count"] > upper].sort_values("message_count", ascending=False)
    anomalies.to_csv(output_dir / "activity_anomalies.csv", index=False)

    forecast_frame = make_forecast_frame(valid.rename(columns={"_parsed_date": "analysis_date"}),
                                         "employee_id", "analysis_date", text_col)
    metrics, predictions = evaluate_forecast(forecast_frame, "employee_id")
    (output_dir / "model_evaluation.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    predictions.to_csv(output_dir / "model_predictions.csv", index=False)
    _save_charts(frame, monthly, "sentiment", visualizations_dir, employee_col="employee")

    classified = int(counts.sum())
    valid_dates = valid["_parsed_date"].dropna()
    report = ["# Employee Sentiment Analysis Report", "", "## Data coverage",
              f"- Source rows: {len(raw)}; valid employee/date rows: {len(valid)}.",
              f"- Date range: {valid_dates.min() if len(valid_dates) else 'unavailable'} to {valid_dates.max() if len(valid_dates) else 'unavailable'}.",
              f"- Rows with invalid dates: {int(frame['_parsed_date'].isna().sum())}; missing employee values: {int(frame[employee_col].isna().sum())}.",
              f"- Predicted sentiment labels: Positive={int(counts['Positive'])}, Neutral={int(counts['Neutral'])}, Negative={int(counts['Negative'])}; unclassified messages={len(frame)-classified}.",
              "", "## Analysis", f"- Employee-months scored: {len(monthly)}.",
              f"- Rolling 30-day qualifying negative-message intervals: {len(risks)}.",
              f"- Activity outlier employee-months by the 1.5×IQR rule: {len(anomalies)}.",
              f"- Forecast evaluation: MAE={metrics['MAE']}, MSE={metrics['MSE']}, RMSE={metrics['RMSE']}, R²={metrics['R2']}.",
              f"- Forecast status: {metrics['note']}", "", "## Interpretation limits",
              f"- Sentiment labels are predictions from `{model_name}`, not ground truth. The model is trained on social-media language and may misread workplace context.",
              "- Flight-risk intervals are the requested negative-message pattern, not proof of intent to resign. Use confidential human review.",
              "- Forecast is an exploratory baseline. The target is the next observed active month, and the holdout is chronological.",
              "- Summary outputs use pseudonymous IDs. The labeled message CSV includes message text and must remain private."]
    (output_dir / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("Analysis complete; artifacts saved to %s and %s", output_dir, visualizations_dir)
    return {"profile": profile, "columns": {"text": text_col, "employee": employee_col, "date": date_col},
            "sentiment_counts": counts.to_dict(), "monthly_scores": monthly, "positive_rankings": positive,
            "negative_rankings": negative, "flight_risks": risks, "forecast_metrics": metrics,
            "activity_anomalies": anomalies, "message_scores": scored_messages,
            "validation": {"monthly_score_reconciliation": True, "ranking_limit": True,
                           "rolling_window_constraints": True}, "identity_count": len(identity_map)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze employee message sentiment.")
    parser.add_argument("--data", default="test.csv", help="Path to the actual CSV input")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--visualizations-dir", default="visualizations")
    parser.add_argument("--text-col")
    parser.add_argument("--employee-col")
    parser.add_argument("--date-col")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run_analysis(args.data, args.output_dir, args.visualizations_dir, args.text_col,
                     args.employee_col, args.date_col, args.model)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("Analysis stopped: %s", exc)
        raise SystemExit(2) from exc
    except Exception:
        logger.exception("Analysis failed unexpectedly")
        raise


if __name__ == "__main__":
    main()
