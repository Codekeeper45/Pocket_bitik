import base64
import hashlib
import unittest
from io import BytesIO

from PIL import Image

from gen_provider import GATEWAY_PROFILE, ProviderProfile
from gen_references import Reference, provider_inputs


def png(color=(20, 40, 60), size=(16, 16)):
    out = BytesIO()
    Image.new("RGB", size, color).save(out, "PNG")
    return out.getvalue()


class ProviderReferenceBoundaryTests(unittest.TestCase):
    def test_packs_all_inputs_in_order_without_mutation(self):
        raws = [png((i, 20, 30)) for i in range(3)]
        encoded = [base64.b64encode(x).decode() for x in raws]
        original = list(encoded)
        refs = [Reference(f"r{i}", 1, i + 1, hashlib.sha256(raw).hexdigest(), raw) for i, raw in enumerate(raws)]
        prompt, packed = provider_inputs("keep REF #1 then REF #2", encoded, GATEWAY_PROFILE, references=refs)
        self.assertEqual(encoded, original)
        self.assertEqual(prompt, "keep REF #1 then REF #2")
        self.assertEqual(len(packed), 3)
        hashes = [hashlib.sha256(base64.b64decode(item.split(",", 1)[1])).hexdigest() for item in packed]
        self.assertEqual(hashes, [r.digest for r in refs])
        self.assertEqual([item.split(":", 1)[1].split(";", 1)[0] for item in packed], ["image/png"] * 3)

    def test_corrupt_base64_and_bool_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            provider_inputs("p", ["%%%"], GATEWAY_PROFILE)
        bad = ProviderProfile("bad", max_bytes=10000, max_images=True)
        with self.assertRaisesRegex(ValueError, "max_images"):
            provider_inputs("p", [base64.b64encode(png()).decode()], bad)

    def test_too_many_images_rejected_without_dropping(self):
        profile = ProviderProfile("small", max_bytes=4_500_000, max_images=2)
        values = [base64.b64encode(png((n, 1, 2))).decode() for n in range(3)]
        with self.assertRaisesRegex(ValueError, "image count"):
            provider_inputs("p", values, profile)

    def test_prompt_is_in_expanded_json_budget(self):
        image = base64.b64encode(png()).decode()
        profile = ProviderProfile("tight", max_bytes=1000, max_images=16)
        with self.assertRaisesRegex(ValueError, "payload exceeds"):
            provider_inputs("x" * 2000, [image], profile)

    def test_oversized_image_rejected(self):
        raw = png(size=(1000, 1000))
        profile = ProviderProfile("tiny", max_bytes=1024, max_images=16)
        with self.assertRaisesRegex(ValueError, "image exceeds byte limit"):
            provider_inputs("p", [base64.b64encode(raw).decode()], profile)

    def test_reference_count_mismatch_rejected(self):
        ref = Reference("r", 1, 1, "0", b"")
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            provider_inputs("p", [base64.b64encode(png()).decode()], GATEWAY_PROFILE, references=[ref, ref])


if __name__ == "__main__":
    unittest.main()
