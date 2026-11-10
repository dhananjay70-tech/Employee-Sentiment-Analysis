# Employee Sentiment Analysis

Python project for classifying employee messages, exploring monthly sentiment, ranking employees by monthly scores, identifying true rolling 30-day negative-message intervals, and evaluating a leakage-aware Linear Regression forecast.

## Data status

The project does not contain an actual `test.csv` dataset. The existing `test.csv.html` is a saved SharePoint viewer page and is not accepted or parsed as input. The notebook and CLI require a real CSV file; no schema or results are assumed. After placing the authorized CSV at the project root, inspect its headers in the notebook and select the actual text, employee, and date columns. A unique header-hint match may be resolved automatically; ambiguous matches stop with an actionable error.

## Project layout

- `notebooks/employee_sentiment_analysis.ipynb` — documented end-to-end entry point, integrated with all `src/` modules.
- `src/sentiment.py` — schema inspection, column resolution, and Hugging Face sentiment labeling.
- `src/scoring.py` — monthly message scores and per-month positive/negative rankings.
- `src/flight_risk.py` — rolling elapsed-time negative-message intervals.
- `src/modeling.py` — next-active-month feature engineering and chronological Linear Regression evaluation.
- `src/analysis.py` — orchestration, validation, reporting, chart creation, and CLI.
- `tests/` — deterministic rule tests for scoring, ranking, and rolling-window boundaries.
- `outputs/` — generated CSVs, evaluation JSON, schema profile, and analysis report.
- `visualizations/` — saved PNG charts.

## Setup (Windows PowerShell)

Use Python 3.11 or later. From the project root, the Python launcher selects an installed Python 3 version:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For macOS/Linux, replace environment creation and activation with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The sentiment model downloads from Hugging Face on first inference. The source file and messages stay local; model download requires network access.

## Run the notebook

1. Put the authorized raw dataset at `test.csv` in the project root.
2. Launch Jupyter from the project root: `jupyter notebook`.
3. Open `notebooks/employee_sentiment_analysis.ipynb` and run all cells in order.
4. Review the printed schema and candidate headers. Set `TEXT_COL`, `EMPLOYEE_COL`, and `DATE_COL` to exact column names if the unique-header resolver cannot determine them.

## Run the command line

After reviewing the actual CSV headers, pass the corresponding fields explicitly:

```powershell
python -m src.analysis --data test.csv --text-col "<actual text column>" --employee-col "<actual employee column>" --date-col "<actual date column>"
```

Omit a field option only if the actual header has one unambiguous match. The CLI logs progress and raises clear errors for missing files, non-CSV input, ambiguous schema, invalid columns, or model failures.

## Tests and static checks

Run validation tests:

```powershell
python -m unittest discover -s tests -v
```

Compile source modules without running the data pipeline:

```powershell
python -m compileall -q src tests
```

Check notebook structure and code cell syntax:

```powershell
python -c "import ast,json; n=json.load(open('notebooks/employee_sentiment_analysis.ipynb',encoding='utf-8')); [ast.parse(''.join(c['source'])) for c in n['cells'] if c['cell_type']=='code']; print('Notebook structure and code cells OK')"
```

## Analysis definitions

- Sentiment classes are Positive, Neutral, and Negative from `cardiffnlp/twitter-roberta-base-sentiment-latest`. Outputs are weak model labels, not ground truth; this model was trained on social-media language and may misread workplace email, sarcasm, and quoted threads.
- Message scores are Positive `+1`, Neutral `0`, Negative `-1`. Employee/month totals use calendar months; message running totals are ordered by timestamp and restart for each employee/month.
- Each month has separate top-three positive and bottom-three negative ranking tables. Ties sort by employee ID alphabetically. Zero totals are not ranked.
- A flight-risk interval includes four or more negative messages in at most 30 elapsed days. It crosses calendar boundaries. Every minimal qualifying four-message window is retained (overlapping intervals are not merged).
- Forecast target: the next observed active calendar month's employee sentiment score. The model uses earlier activity/message-length fields and sentiment history. Train/test split is chronological by target month; metrics are MAE, MSE, RMSE, and R² where defined.
- Unusual activity uses a descriptive 1.5×IQR rule over employee/month message counts; it is not a risk diagnosis.

## Outputs

All generated tables are saved to `outputs/`: `schema_profile.json`, `labeled_messages.csv`, `message_scores.csv`, `monthly_employee_scores.csv`, `monthly_top_positive.csv`, `monthly_top_negative.csv`, `flight_risk_report.csv`, `activity_anomalies.csv`, `model_predictions.csv`, `model_evaluation.json`, and `analysis_report.md`. Charts are saved to `visualizations/` as sentiment distribution, monthly activity, monthly sentiment trend, and monthly employee score heatmap PNGs.

The labeled message CSV necessarily contains message text for review and can contain confidential content. Summary outputs use stable pseudonymous IDs and do not export the source identity column. Keep `outputs/`, `visualizations/`, and raw files private. `.gitignore` excludes CSVs, the known HTML viewer export, and generated output files; verify repository status before publishing.

Sentiment scores do not measure performance, wellbeing, or intent to resign. Flight-risk intervals are only the requested negative-message pattern; use confidential human review and do not treat the flag as a conclusion. Forecast metrics can be unstable with sparse histories and do not establish causality.
