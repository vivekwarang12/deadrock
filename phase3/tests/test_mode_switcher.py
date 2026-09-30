"""
test_mode_switcher.py  –  Phase 3 Mode-Switching Latency Tests
==============================================================
Validates that the seamless mode-switching state machine:
  1. Detects GNSS loss and transitions to PURE_INERTIAL_DR within 1 sample (< 10 ms at 100Hz).
  2. Applies hysteresis to prevent chattering under noisy reception.
"""

import unittest
from phase3.src.step3_mode_switcher import ModeSwitcher, ModeSwitcherConfig, NavigationMode


class TestModeSwitcher(unittest.TestCase):

    def setUp(self):
        self.cfg = ModeSwitcherConfig(
            max_accuracy_m=10.0,
            min_satellites=4,
            max_hdop=3.0,
            hysteresis_count=3,
            timeout_seconds=1.0
        )
        self.switcher = ModeSwitcher(self.cfg)

    def test_nominal_gnss_mode(self):
        # Establish healthy GPS lock
        for i in range(5):
            mode = self.switcher.evaluate_sample(t_sec=i * 0.1, has_gnss=True, accuracy_m=3.0)
        self.assertEqual(mode, NavigationMode.GNSS_AIDED_INS)

    def test_instant_outage_transition(self):
        # Establish lock
        for i in range(5):
            self.switcher.evaluate_sample(t_sec=i * 0.1, has_gnss=True, accuracy_m=3.0)

        # Drop GPS at t=0.6
        mode = self.switcher.evaluate_sample(t_sec=0.6, has_gnss=False)
        self.assertEqual(mode, NavigationMode.PURE_INERTIAL_DR)

    def test_hysteresis_recovery(self):
        # Outage
        self.switcher.evaluate_sample(t_sec=1.0, has_gnss=False)
        self.assertEqual(self.switcher.current_mode, NavigationMode.PURE_INERTIAL_DR)

        # 1 good sample should NOT immediately restore full GNSS mode (needs 3 consecutive)
        m1 = self.switcher.evaluate_sample(t_sec=1.1, has_gnss=True, accuracy_m=3.0)
        self.assertNotEqual(m1, NavigationMode.GNSS_AIDED_INS)

        m2 = self.switcher.evaluate_sample(t_sec=1.2, has_gnss=True, accuracy_m=3.0)
        m3 = self.switcher.evaluate_sample(t_sec=1.3, has_gnss=True, accuracy_m=3.0)
        self.assertEqual(m3, NavigationMode.GNSS_AIDED_INS)


if __name__ == '__main__':
    unittest.main()
