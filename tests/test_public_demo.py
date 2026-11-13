"""Small offline acceptance suite for the public synthetic UI only."""
import os
import socket
import unittest
from unittest.mock import patch
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from src.yuqing.serve.research_history import ResearchHistory


class PublicDemoTests(unittest.TestCase):
    """Check fixtures and the absence of investigation endpoints without network access."""

    def test_success_projection(self):
        """Show three mock roles and distinct E/W quotes with no real links."""
        view = ResearchHistory(ROOT / "examples/synthetic/history").load("1" * 32)
        self.assertIn("Mock", view[0])
        self.assertEqual(view[1].count("completed"), 3)
        self.assertIn("E1", view[3])
        self.assertIn("W1", view[3])
        self.assertNotIn("href=", view[3])
        self.assertIn("合成", view[4])

    def test_failure_retained(self):
        """Keep failed fixtures visible without inventing a completed result."""
        history = ResearchHistory(ROOT / "examples/synthetic/history")
        choices, selected = history.choices()
        self.assertEqual(len(choices), 2)
        self.assertEqual(selected, "1" * 32)
        self.assertEqual({identity for _, identity in choices}, {"1" * 32, "2" * 32})
        self.assertTrue(all(identity[:8] not in label for label, identity in choices))
        view = history.load("2" * 32)
        self.assertIn("failed", view[0])
        self.assertNotIn("completed", view[1])
        self.assertIn("未生成事实结论", view[4])

    def test_readonly_ui_no_network(self):
        """Verify UI-only imports in a fresh interpreter, independent of pytest collection."""
        import subprocess
        result = subprocess.run([sys.executable, "-B", str(ROOT / "tests/public_demo_probe.py")],
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("readonly_ui_verified", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
