import base64
import hashlib
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image

from gen_references import make_reference
from gen_runtime import CURRENT, Invocation


def png(color):
    out = BytesIO()
    Image.new("RGB", (8, 8), color).save(out, "PNG")
    return out.getvalue()


class ActualReferenceRegistryTests(unittest.TestCase):
    def test_registry_carries_real_images_and_remaps_duplicates(self):
        import bot_new
        first, catalog_img, duplicate = png((2, 3, 4)), png((5, 6, 7)), png((2, 3, 4))
        invocation = Invocation()
        token = CURRENT.set(invocation)
        try:
            result = bot_new._gen_remap_selected_refs(
                [base64.b64encode(first).decode()], ["submitted"],
                [{"idx": 7, "chat": -1005, "mid": 33, "bytes": catalog_img, "caption": "caption"},
                 {"idx": 8, "chat": -1005, "mid": 34, "bytes": duplicate}],
                [(7, "primary"), (8, "matching")] )
        finally:
            CURRENT.reset(token)
        inputs, roles, selected, mapping = result
        self.assertEqual([hashlib.sha256(base64.b64decode(x)).hexdigest() for x in inputs],
                         [hashlib.sha256(first).hexdigest(), hashlib.sha256(catalog_img).hexdigest()])
        self.assertEqual(selected, [7, 8])
        self.assertEqual(mapping, {7: 2, 8: 1})
        self.assertEqual(roles, [(1, "subject"), (2, "subject")])
        refs = invocation.timings["reference_registry"]["references"]
        self.assertEqual([r["provenance"][0]["source_type"] for r in refs], ["submitted", "catalog"])
        self.assertEqual(len(refs[0]['provenance']), 2)
        self.assertNotIn("owner", refs[0])
        origin = refs[1]['provenance'][0]
        self.assertEqual((origin["chat_id"], origin["message_id"], origin["caption"]), (-1005, 33, "caption"))
        self.assertEqual(refs[0]["mime"], "image/png")

    def test_missing_or_invalid_selected_ref_does_not_block_fallback(self):
        import bot_new
        self.assertEqual(bot_new._gen_remap_selected_refs([], [], [], [(99, "subject")])[0], [])
        huge = b"x" * (bot_new.GEN_IMAGE_MAX_INPUT + 1)
        self.assertEqual(bot_new._gen_remap_selected_refs([], [], [{"idx": 1, "bytes": huge}], [(1, "subject")])[0], [])

    def test_make_reference_records_hash_mime_and_size(self):
        raw = png((7, 8, 9))
        ref = make_reference(-1001, 12, raw)
        self.assertEqual(ref.digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(ref.mime_type, "image/png")
        self.assertEqual(ref.size_bytes, len(raw))
        self.assertEqual(ref.source_type, "catalog")


if __name__ == "__main__":
    unittest.main()
