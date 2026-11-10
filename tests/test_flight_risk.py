import unittest
import pandas as pd

from src.flight_risk import find_flight_risks


class RollingFlightRiskTests(unittest.TestCase):
    def test_detects_four_negative_emails_across_calendar_months(self):
        rows = pd.DataFrame({
            "person": ["A"] * 4,
            "sent_at": ["2025-01-20", "2025-01-29", "2025-02-05", "2025-02-19"],
            "sentiment": ["Negative"] * 4,
        })
        risks = find_flight_risks(rows, "person", "sent_at")
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks.iloc[0]["negative_email_count"], 4)
        self.assertEqual(risks.iloc[0]["window_end"] - risks.iloc[0]["window_start"], pd.Timedelta(30, unit="D"))

    def test_excludes_fourth_negative_outside_30_elapsed_days(self):
        rows = pd.DataFrame({
            "person": ["A"] * 4,
            "sent_at": ["2025-01-01", "2025-01-10", "2025-01-20", "2025-02-01"],
            "sentiment": ["Negative"] * 4,
        })
        self.assertTrue(find_flight_risks(rows, "person", "sent_at").empty)

    def test_ignores_positive_neutral_and_other_employee_messages(self):
        rows = pd.DataFrame({
            "person": ["A", "A", "A", "B", "A", "A"],
            "sent_at": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"],
            "sentiment": ["Negative", "Positive", "Negative", "Negative", "Neutral", "Negative"],
        })
        self.assertTrue(find_flight_risks(rows, "person", "sent_at").empty)

    def test_invalid_threshold_fails_loudly(self):
        with self.assertRaises(ValueError):
            find_flight_risks(pd.DataFrame(columns=["person", "sent_at", "sentiment"]), "person", "sent_at", threshold=0)


if __name__ == "__main__":
    unittest.main()
