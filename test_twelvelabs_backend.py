"""
Tests for the optional TwelveLabs Pegasus transcription backend.

Run with:
    python -m unittest test_twelvelabs_backend

The parsing tests run offline and always execute. The live smoke test only
runs when TWELVELABS_API_KEY is set and the `twelvelabs` package is installed;
it is skipped otherwise so CI without a key still passes.
"""

import os
import unittest

import twelvelabs_backend as tlb


class TestParsing(unittest.TestCase):
    """No-network tests for the output parsers and result shape."""

    def test_parse_ts_variants(self):
        self.assertAlmostEqual(tlb._parse_ts("00:00:01.500"), 1.5)
        self.assertAlmostEqual(tlb._parse_ts("01:02"), 62.0)
        self.assertAlmostEqual(tlb._parse_ts("00:01:02"), 62.0)
        self.assertAlmostEqual(tlb._parse_ts("5"), 5.0)
        self.assertIsNone(tlb._parse_ts(""))
        self.assertIsNone(tlb._parse_ts("not-a-time"))

    def test_segments_from_bracketed_text(self):
        raw = (
            "[00:00:00.000 --> 00:00:02.500] Hello there.\n"
            "[00:00:02.500 --> 00:00:05.000] How are you?"
        )
        segs = tlb._segments_from_pegasus_text(raw)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["text"], "Hello there.")
        self.assertAlmostEqual(segs[0]["start"], 0.0)
        self.assertAlmostEqual(segs[0]["end"], 2.5)
        self.assertEqual(segs[1]["text"], "How are you?")

    def test_segments_keeps_untimestamped_lines(self):
        segs = tlb._segments_from_pegasus_text("Just a paragraph with no times.")
        self.assertEqual(len(segs), 1)
        self.assertIsNone(segs[0]["start"])
        self.assertEqual(segs[0]["text"], "Just a paragraph with no times.")

    def test_normalize_raw_passthrough_and_paragraph(self):
        bracketed = "[00:00:00.000 --> 00:00:01.000] Hi"
        self.assertEqual(tlb._normalize_raw(bracketed), bracketed)
        self.assertEqual(tlb._normalize_raw("  a paragraph  "), "a paragraph")
        self.assertEqual(tlb._normalize_raw(""), "")

    def test_resolve_api_key_missing_raises(self):
        saved = os.environ.pop("TWELVELABS_API_KEY", None)
        try:
            with self.assertRaises(RuntimeError):
                tlb._resolve_api_key({})
            self.assertEqual(tlb._resolve_api_key({"twelvelabs_api_key": "abc"}), "abc")
        finally:
            if saved is not None:
                os.environ["TWELVELABS_API_KEY"] = saved

    def test_result_shape_matches_whisper_contract(self):
        r = tlb._result("raw", "text", [], 1.0, "", False)
        self.assertEqual(
            set(r.keys()),
            {"raw", "text", "segments", "audio_length", "stderr", "cancelled"},
        )


@unittest.skipUnless(
    os.environ.get("TWELVELABS_API_KEY"), "TWELVELABS_API_KEY not set"
)
class TestLiveSmoke(unittest.TestCase):
    """Live wiring check against the TwelveLabs API (requires a key)."""

    def test_client_constructs_and_analyze_is_callable(self):
        from twelvelabs import TwelveLabs

        client = TwelveLabs(api_key=os.environ["TWELVELABS_API_KEY"])
        self.assertTrue(callable(client.analyze))
        self.assertTrue(hasattr(client, "assets"))


if __name__ == "__main__":
    unittest.main()
