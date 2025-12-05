import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load your reference (training) and current (hold-out or recent prod batch)
ref = pd.read_csv("data/raw/amazon.csv")
cur = pd.read_csv("data/raw/amazon.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

# Save static HTML report
report.save_html("monitoring/evidently_report.html")
print("Saved: monitoring/evidently_report.html")
