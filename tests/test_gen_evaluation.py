import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from gen_evaluation import (
    EvaluationError, ExperimentOptions, candidate_count_for, inspect_required_text,
    measured_upscale, summarize, validate_manifest, validate_profile_options,
)

ROOT = pathlib.Path(__file__).parent


class EvaluationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((ROOT / "fixtures/gen_eval/manifest.json").read_text())

    def test_fixture_is_valid_and_aggregates_measured_evidence(self):
        result = summarize(self.manifest)
        self.assertEqual(result["runs"], 3)
        self.assertEqual(result["variants"]["single"]["n"], 2)
        self.assertAlmostEqual(result["variants"]["single"]["mean_quality"]["fidelity"], .7)
        self.assertAlmostEqual(result["variants"]["single"]["mean_cost_usd"], .045)
        self.assertEqual(result["variants"]["two-candidates"]["mean_calls"]["generation"], 2)

    def test_invalid_manifest_evidence_fails_closed(self):
        for mutate in (
            lambda d: d.update(schema_version=2),
            lambda d: d["runs"][0].update(cost_usd=-1),
            lambda d: d["runs"][0].update(quality={"fidelity": 1.2}),
            lambda d: d["runs"][0].update(timing_ms={}),
            lambda d: d["runs"][0].update(artifacts=[]),
        ):
            candidate = json.loads(json.dumps(self.manifest))
            mutate(candidate)
            with self.assertRaises(EvaluationError):
                validate_manifest(candidate)

    def test_candidate_sampling_disabled_by_default(self):
        self.assertEqual(candidate_count_for(ExperimentOptions()), 1)
        with self.assertRaises(EvaluationError):
            ExperimentOptions(candidate_count=2)
        self.assertEqual(candidate_count_for(ExperimentOptions(2, True)), 2)

    def test_ocr_adapter_is_dependency_injected_and_optional(self):
        self.assertEqual(inspect_required_text("HELLO", "unused.png").adapter, "unavailable")
        result = inspect_required_text("HELLO", "unused.png", lambda _: "HELLO")
        self.assertTrue(result.exact_match)
        with self.assertRaises(EvaluationError):
            inspect_required_text("HELLO", "unused.png", lambda _: None)

    def test_upscale_reports_measured_dimensions_without_detail_claim(self):
        report = measured_upscale((512, 512), (1024, 1024), method="synthetic-fixture-nearest")
        self.assertEqual(report["source_dimensions"], [512, 512])
        self.assertFalse(report["detail_improvement_claimed"])
        self.assertTrue(report["dimensions_measured"])

    def test_mask_option_requires_declared_profile_capability(self):
        validate_profile_options(SimpleNamespace(supports_masks=False))
        with self.assertRaises(EvaluationError):
            validate_profile_options(SimpleNamespace(supports_masks=False), use_mask=True)
        validate_profile_options(SimpleNamespace(supports_masks=True), use_mask=True)


if __name__ == "__main__":
    unittest.main()


def test_manifest_cli_smoke():
    """Executable CLI smoke test uses synthetic evidence only."""
    import subprocess, sys
    proc = subprocess.run([sys.executable, str(ROOT.parent / "gen_evaluation.py"), str(ROOT / "fixtures/gen_eval/manifest.json")],
                          capture_output=True, text=True, check=True)
    assert '"runs": 3' in proc.stdout
    assert '"mean_cost_usd": 0.045' in proc.stdout


if __name__ == "__main__":
    unittest.main()
