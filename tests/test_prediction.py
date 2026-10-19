
import unittest

from src.preprocessing import (
    EXPECTED_FEATURES,
    build_input_frame,
    get_aqi_category,
)


class TestPredictionInput(unittest.TestCase):

    def setUp(self):
        self.values = {
            feature: 1.0 for feature in EXPECTED_FEATURES
        }

    def test_input_has_one_row_and_correct_order(self):
        frame = build_input_frame(self.values)

        self.assertEqual(frame.shape, (1, 12))
        self.assertEqual(list(frame.columns), EXPECTED_FEATURES)

    def test_missing_feature_is_rejected(self):
        values = dict(self.values)
        del values["PM2.5"]

        with self.assertRaisesRegex(
            ValueError, "Missing pollutant inputs"
        ):
            build_input_frame(values)

    def test_negative_value_is_rejected(self):
        values = dict(self.values)
        values["NO2"] = -1

        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_input_frame(values)

    def test_aqi_categories(self):
        self.assertEqual(get_aqi_category(30), "Good")
        self.assertEqual(get_aqi_category(75), "Satisfactory")
        self.assertEqual(get_aqi_category(150), "Moderate")
        self.assertEqual(get_aqi_category(250), "Poor")
        self.assertEqual(get_aqi_category(350), "Very Poor")
        self.assertEqual(get_aqi_category(450), "Severe")
        self.assertEqual(get_aqi_category(501), "Unclassified")


if __name__ == "__main__":
    unittest.main()
