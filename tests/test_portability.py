from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAPERS = [
    ROOT / "docs" / "result" / "第二问求解与结果分析.md",
    ROOT / "outputs" / "q2_review_v2" / "第二问求解与结果分析.md",
    ROOT / "outputs" / "q2_review_v2" / "第二问原版论文_存档.md",
    ROOT / "docs" / "第二问场景自适应模型说明.md",
]


def _markdown_targets(text):
    return [target.split("#", 1)[0] for target in re.findall(r"\]\(([^)]+)\)", text)]


class PortabilityTests(unittest.TestCase):
    def test_q2_markdown_links_are_relative_and_resolve(self):
        for paper in PAPERS:
            text = paper.read_text(encoding="utf-8")
            self.assertNotIn("C:" + chr(47) + "Users" + chr(47), text)
            self.assertNotIn("C:" + "\\Users\\", text)
            for target in _markdown_targets(text):
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                self.assertFalse(Path(target).is_absolute(), (paper, target))
                self.assertTrue((paper.parent / target).resolve().exists(), (paper, target))


    def test_dependency_manifest_covers_q2_runtime(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for package in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl"):
            self.assertIn(package, requirements)


    def test_gitignore_keeps_delivery_figures_and_ignores_previews(self):
        def ignored(path):
            return subprocess.run(
                ["git", "check-ignore", "-q", path], cwd=ROOT, check=False
            ).returncode == 0

        self.assertTrue(ignored("outputs/q2_review_v2/result_计划购电量.png"))
        self.assertTrue(ignored("outputs/q2_review_v2/result2.xlsx.inspect.ndjson"))
        self.assertFalse(ignored("outputs/q2_review_v2/monthly_cost_v2.png"))
        self.assertFalse(ignored("outputs/q2_review_v2/result2.xlsx"))
