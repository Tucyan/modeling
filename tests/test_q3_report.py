"""Behavior tests for the generated Question 3 paper section."""
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p)
                for p in ('src',)]

from q3_report import render_report


def minimal_payload():
    return {
        'winner': 'a0_0.85_au_0.75__R0',
        'selected': {'risk_pair': {'alpha0': .85, 'alpha_update': .75},
                     'release_name': 'R0', 'updates': []},
        'comparison': [
            {'label': 'calibrated_R0', 'settlement': 'N0', 'cost': 100.,
             'emergency_kwh': 2., 'emergency_days': 1, 'emergency_events': 1,
             'plan_cost': 90., 'up_cost': 0., 'down_fee': 0.,
             'emergency_cost': 10., 'unused_kwh': 3., 'spill_kwh': 4.},
            {'label': 'calibrated_Rall', 'settlement': 'N0', 'cost': 130.,
             'emergency_kwh': 0., 'emergency_days': 0, 'emergency_events': 0,
             'plan_cost': 90., 'up_cost': 40., 'down_fee': 0.,
             'emergency_cost': 0., 'unused_kwh': 8., 'spill_kwh': 4.},
        ],
        'specified': {}, 'q2_cost': 95.,
        'calibration_start_day': 14, 'calibration_end_exclusive': 31,
        'last_history_day': 30, 'decision_day': 31,
    }


class Q3ReportTests(unittest.TestCase):
    def test_report_states_causal_selection_and_settlement_scope(self):
        """The paper must not hide the information boundary or mix settlement rules."""
        text = render_report(minimal_payload())
        self.assertIn('2025-01-15 至 2025-01-31', text)
        self.assertNotIn('1月14—30日', text)
        self.assertIn('N0', text)
        self.assertIn('不退款', text)
        self.assertIn('未来实际', text)
        self.assertIn('result3.xlsx', text)

    def test_report_does_not_claim_every_release_is_necessary(self):
        """A rejected intraday release set cannot be described as necessary."""
        text = render_report(minimal_payload())
        self.assertIn('并非每个发布时间', text)
        self.assertIn('仅0点', text)

    def test_intraday_winner_does_not_receive_r0_conclusions(self):
        payload = minimal_payload()
        payload['selected'].update(release_name='Rall', updates=[6, 12, 18])
        payload['comparison'][1].update(plan_cost=70., up_cost=10., cost=80.,
                                        unused_kwh=2., spill_kwh=1.)
        text = render_report(payload)
        self.assertIn('采用所选的日内调整', text)
        self.assertNotIn('不需要据此修改正式购电合同', text)
        self.assertNotIn('选中的 R0 策略', text)
        self.assertNotIn('额外合同修订未覆盖其成本', text)
        self.assertIn('减少 20.00 元', text)

    def test_r0_selection_is_distinguished_from_better_hindsight_policy(self):
        payload = minimal_payload()
        payload['comparison'][1]['cost'] = 80.
        text = render_report(payload)
        self.assertIn('事后对照', text)
        self.assertIn('减少 20.00 元', text)
        self.assertNotIn('额外合同修订未覆盖其成本', text)
        self.assertNotIn('不需要据此修改正式购电合同', text)

    def test_missing_refund_experiment_is_reported_as_missing(self):
        text = render_report(minimal_payload())
        self.assertIn('未提供', text)
        self.assertNotIn('给出相同费用', text)

    def test_refund_comparison_uses_actual_selected_policy_result(self):
        payload = minimal_payload()
        payload['selected'].update(release_name='Rall', updates=[6, 12, 18])
        payload['comparison'].append({'label': 'N1_Rall', 'cost': 120.})
        text = render_report(payload)
        self.assertIn('120.00 元', text)
        self.assertIn('减少 10.00 元', text)
        self.assertNotIn('选中的 R0 策略', text)

    def test_equal_costs_are_not_described_as_a_saving(self):
        payload = minimal_payload()
        payload['selected'].update(release_name='Rall', updates=[6, 12, 18])
        payload['comparison'][1]['cost'] = 100.
        text = render_report(payload)
        self.assertIn('费用持平', text)
        self.assertNotIn('减少 0.00 元', text)

    def test_protocol_dates_are_not_hardcoded(self):
        payload = minimal_payload()
        payload.update(calibration_start_day=13, calibration_end_exclusive=30,
                       last_history_day=29, decision_day=31)
        text = render_report(payload)
        self.assertIn('2025-01-14 至 2025-01-30', text)
        self.assertNotIn('2025-01-15 至 2025-01-31', text)


if __name__ == '__main__':
    unittest.main()
