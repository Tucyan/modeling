import unittest
from pathlib import Path
class ReportTests(unittest.TestCase):
    def test_no_fixed_false_event_statement(self):
        source=(Path(__file__).resolve().parents[1]/'src/q2_report.py').read_text(encoding='utf8')
        self.assertNotIn('四个指定日的紧急购电均为零',source)
        self.assertNotIn('有104天发生紧急购电',source)
if __name__=='__main__': unittest.main()
