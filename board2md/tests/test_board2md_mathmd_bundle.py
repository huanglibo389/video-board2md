from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import board2md_core as MODULE  # noqa: E402


class FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "# 几何题\n\n## 题面\n\n证明 $AB=AC$。\n\n## 解答\n\n连接 $AD$。"
                    }
                }
            ]
        }


class Board2MdMathMdBundleTests(unittest.TestCase):
    def test_keyless_local_run_can_emit_canonical_bundle_with_source_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "board page.png"
            image.write_bytes(b"image")
            bundle = root / "bundle"
            output = root / "board.md"
            calls = []

            def fake_post(**kwargs):
                calls.append(kwargs)
                return FakeResponse()

            code = MODULE.main(
                [
                    str(image),
                    "--output",
                    str(output),
                    "--bundle-dir",
                    str(bundle),
                    "--bundle-fuse",
                    "never",
                ],
                request_post=fake_post,
                env={},
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("Authorization", calls[0]["headers"])
            problem = next((bundle / "problems").glob("p*.md"))
            content = problem.read_text(encoding="utf-8")
            self.assertIn("## 参考图像", content)
            self.assertIn("assets/", content)
            manifest = json.loads((bundle / "mathmd_manifest.json").read_text(encoding="utf-8"))
            source = manifest["documents"][0]["sources"][0]
            self.assertEqual(source["kind"], "board")
            self.assertEqual(source["source"], str(image))
            self.assertEqual(manifest["quality_summary"]["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
