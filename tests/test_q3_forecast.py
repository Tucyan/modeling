"""Causal forecast-release and risk-reserve tests for Question 3."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src',)]
import numpy as np

import pandas as pd
from q3_forecast import pv_release_10min, residual_samples, risk_reserve
from q3_solver import pv_at_issue


def synthetic_inputs(days=20):
    slots = np.arange(144, dtype=float)
    actual_load = np.vstack([100 + day + slots / 100 for day in range(days)])
    actual_pv = np.vstack([np.maximum(0, 30 - abs(slots - 72) / 3)
                           for _ in range(days)])
    load_forecast = actual_load - 2
    hourly = np.zeros((days, 4, 24), dtype=float)
    for day in range(days):
        for release in range(4):
            clock_profile = actual_pv[day].reshape(24, 6).mean(axis=1) * 6
            hourly[day, release] = np.roll(clock_profile, -release * 6)
    return actual_load, actual_pv, load_forecast, hourly


class Q3ForecastTests(unittest.TestCase):
    def test_release_horizons_start_at_one_for_every_issue(self):
        hourly = np.tile(np.arange(1., 25.) * 6, (20, 4, 1))
        for hour in (0, 6, 12, 18):
            with self.subTest(hour=hour):
                expected = np.repeat(np.arange(1., 25. - hour), 6)
                np.testing.assert_array_equal(pv_release_10min(hourly, 15, hour), expected)
                np.testing.assert_array_equal(pv_at_issue(hourly, 15, hour), expected)

    def test_next_day_forecasts_do_not_change_todays_remainder(self):
        hourly = np.ones((20, 4, 24)) * 6
        before = pv_release_10min(hourly, 15, 18)
        hourly[15, 3, 6:] = 99999
        np.testing.assert_array_equal(pv_release_10min(hourly, 15, 18), before)

    def test_perfect_release_forecasts_produce_zero_net_residual(self):
        profile = np.maximum(0., 12 - abs(np.arange(24) - 12))
        pv = np.tile(np.repeat(profile, 6), (20, 1))
        load = np.full((20, 144), 50.)
        hourly = np.empty((20, 4, 24))
        for issue, hour in enumerate((0, 6, 12, 18)):
            hourly[:, issue] = np.roll(profile, -hour) * 6
        for hour in (0, 6, 12, 18):
            with self.subTest(hour=hour):
                samples = residual_samples(15, hour, load, pv, load, hourly)
                np.testing.assert_allclose(samples, 0, atol=1e-12)

    def test_real_release_mapping_matches_standardized_time_windows(self):
        root = Path(__file__).resolve().parents[1]
        data = pd.read_csv(root / 'C题_标准化数据与修正模板/C_final_delivery/data_csv/pv_forecast_hourly.csv')
        hourly = np.zeros((365, 4, 24))
        for date in ('2025-02-01', '2025-06-21', '2025-12-31'):
            day = (pd.Timestamp(date) - pd.Timestamp('2025-01-01')).days
            for hour in (0, 6, 12, 18):
                issue = data[(data.issue_date == date) & (data.issue_time == f'{hour}:00')]
                issue = issue.sort_values('horizon_hour')
                hourly[day, hour // 6] = issue.pv_forecast_kw.to_numpy()
                start = pd.to_datetime(issue.forecast_window_start)
                end = pd.to_datetime(issue.forecast_window_end)
                remaining = issue[(start >= pd.Timestamp(date) + pd.Timedelta(hours=hour)) &
                                  (end <= pd.Timestamp(date) + pd.Timedelta(days=1))]
                expected = np.repeat(remaining.pv_forecast_kw.to_numpy(), 6) / 6
                with self.subTest(date=date, hour=hour):
                    np.testing.assert_array_equal(pv_release_10min(hourly, day, hour), expected)
                    np.testing.assert_array_equal(pv_at_issue(hourly, day, hour), expected)

    def test_residual_history_excludes_decision_day_and_future(self):
        """Changing decision-day/future actuals must not change a past-only reserve."""
        actual_load, actual_pv, load_forecast, hourly = synthetic_inputs()
        before = residual_samples(
            15, 6, actual_load, actual_pv, load_forecast, hourly, lookback=8)
        actual_load[15:] += 999999
        after = residual_samples(
            15, 6, actual_load, actual_pv, load_forecast, hourly, lookback=8)
        np.testing.assert_allclose(before, after)
        self.assertEqual(before.shape, (8, 108))

    def test_risk_reserve_is_horizon_specific_quantile(self):
        """Each remaining slot needs its own historical error quantile."""
        samples = np.array([[0., 10.], [10., 20.], [20., 30.]])
        np.testing.assert_allclose(risk_reserve(samples, .5), [10., 20.])

    def test_invalid_release_or_noncausal_history_is_rejected(self):
        """Unsupported release times and an empty causal window are data errors."""
        actual_load, actual_pv, load_forecast, hourly = synthetic_inputs()
        with self.assertRaises(ValueError):
            residual_samples(15, 5, actual_load, actual_pv, load_forecast, hourly)
        with self.assertRaises(ValueError):
            residual_samples(7, 0, actual_load, actual_pv, load_forecast, hourly)

    def test_unused_warmup_forecasts_may_be_missing(self):
        """Missing forecasts before the first eligible history day must be ignored."""
        actual_load, actual_pv, load_forecast, hourly = synthetic_inputs()
        load_forecast[:7] = np.nan
        samples = residual_samples(
            15, 6, actual_load, actual_pv, load_forecast, hourly, lookback=8)
        self.assertEqual(samples.shape, (8, 108))
        self.assertTrue(np.isfinite(samples).all())


if __name__ == '__main__':
    unittest.main()
