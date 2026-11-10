import unittest
import pandas as pd

from src.scoring import monthly_rankings, monthly_scores


class MonthlyScoringTests(unittest.TestCase):
    def test_monthly_cumulative_score_resets_and_uses_date_order(self):
        rows = pd.DataFrame({
            "person": ["A", "A", "A", "A"],
            "sent_at": ["2025-02-01", "2025-01-03", "2025-01-01", "2025-02-02"],
            "sentiment": ["Positive", "Negative", "Positive", "Neutral"],
        })
        messages, monthly = monthly_scores(rows, "person", "sent_at")
        jan = messages[messages["month"].eq("2025-01")]
        feb = messages[messages["month"].eq("2025-02")]
        self.assertEqual(jan["monthly_running_score"].tolist(), [1, 0])
        self.assertEqual(feb["monthly_running_score"].tolist(), [1, 1])
        self.assertEqual(monthly.set_index("month")["monthly_score"].to_dict(), {"2025-01": 0, "2025-02": 1})

    def test_ranking_sorts_by_score_then_employee_name(self):
        monthly = pd.DataFrame({
            "month": ["2025-01"] * 5,
            "person": ["Zed", "Amy", "Bob", "Lee", "Moe"],
            "monthly_score": [3, 3, -3, -3, 0],
            "message_count": [3] * 5,
            "positive_count": [3, 3, 0, 0, 0],
            "neutral_count": [0] * 5,
            "negative_count": [0, 0, 3, 3, 0],
        })
        positive, negative = monthly_rankings(monthly, "person")
        self.assertEqual(positive["person"].tolist(), ["Amy", "Zed"])
        self.assertEqual(negative["person"].tolist(), ["Bob", "Lee"])

    def test_invalid_sentiment_and_missing_columns_fail_loudly(self):
        with self.assertRaises(ValueError):
            monthly_scores(pd.DataFrame({"person": ["A"], "sent_at": ["2025-01-01"], "sentiment": ["Mixed"]}), "person", "sent_at")
        with self.assertRaises(ValueError):
            monthly_scores(pd.DataFrame({"person": ["A"]}), "person", "sent_at")


if __name__ == "__main__":
    unittest.main()
