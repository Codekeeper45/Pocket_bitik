from gen_provider import ProviderProfile
from gen_references import ReferenceRegistry, make_reference
from tests.test_gen_reference_registry import image_bytes
import pytest


def test_full_working_set_registers_inspected_and_protects_user_and_selection():
    user = make_reference(0, 1, image_bytes(), source_type="submitted")
    chosen = make_reference(1, 2, image_bytes("blue"))
    registry = ReferenceRegistry((user, chosen))
    for number in range(8):
        registry.add(make_reference(1, number + 3, image_bytes((number, 1, 2)), source_type="web"))
    inspected = registry.add(make_reference(2, 1, image_bytes("yellow")))
    result = registry.working_set(3, selected_ids=(chosen.ref_id,), inspected_ids=(inspected.ref_id,))
    assert {ref.ref_id for ref in result.refs} == {user.ref_id, chosen.ref_id, inspected.ref_id}
    assert len(registry.refs) == 11
    assert not result.overflow_ids
    overflow = registry.working_set(1, selected_ids=(chosen.ref_id,))
    assert len(overflow.refs) == 1
    assert overflow.overflow_ids
    assert len(registry.refs) == 11


def test_packing_overflow_reports_required_hero_without_losing_registry():
    refs = [make_reference(1, number, image_bytes(color), role="subject")
            for number, color in enumerate(("red", "blue"), 1)]
    registry = ReferenceRegistry(refs)
    result = registry.pack([ref.ref_id for ref in refs], ProviderProfile("one", max_images=1),
                           required_ids=[ref.ref_id for ref in refs], prompt="two heroes")
    assert result.status == "needs_reselection"
    assert len(result.refs) == 1 and result.required_missing
    assert len(registry.refs) == 2
    assert result.prompt == "two heroes"


def test_bad_bytes_and_missing_refs_return_fallback_not_exception():
    bad = make_reference(1, 1, b"not an image")
    registry = ReferenceRegistry((bad,))
    result = registry.pack((bad.ref_id, "missing"), ProviderProfile("test"), prompt="original intent")
    assert result.status == "text_only"
    assert result.refs == () and len(result.issues) == 2
    assert result.prompt == "original intent"


def test_packing_resizes_only_when_needed_and_validates_payload():
    from gen_references import provider_inputs

    ref = make_reference(1, 1, image_bytes(size=(1500, 1500)))
    registry = ReferenceRegistry((ref,))
    roomy = registry.pack((ref.ref_id,), ProviderProfile("roomy"), prompt="intent")
    assert roomy.refs[0].data == ref.data
    tight = ProviderProfile("tight", max_bytes=4000)
    packed = registry.pack((ref.ref_id,), tight, prompt="intent")
    assert packed.refs and len(packed.refs[0].data) < len(ref.data)
    assert packed.refs[0].ref_id == ref.ref_id
    assert packed.refs[0].digest != ref.digest
    provider_inputs(packed.prompt, [item.data for item in packed.refs], tight, references=packed.refs)


def test_required_refs_take_precedence_over_optional_candidates():
    optional = make_reference(1, 1, image_bytes(), priority=100)
    required = make_reference(1, 2, image_bytes("blue"))
    registry = ReferenceRegistry((optional, required))
    result = registry.pack((optional.ref_id, required.ref_id), ProviderProfile("one"),
                           required_ids=(required.ref_id,))
    assert result.refs[0].ref_id == required.ref_id
    assert not result.required_missing


def test_two_required_large_images_are_compacted_together():
    import json
    import random
    from io import BytesIO
    from PIL import Image
    from gen_references import provider_inputs

    refs = []
    for seed in (1, 2):
        stream = BytesIO()
        Image.frombytes("RGB", (256, 256), random.Random(seed).randbytes(256 * 256 * 3)).save(stream, "PNG")
        refs.append(make_reference(1, seed, stream.getvalue(), role="subject"))
    _, encoded = provider_inputs("", [refs[0].data], ProviderProfile("large"))
    budget = len(json.dumps({"prompt": "", "images": encoded}, separators=(",", ":")).encode()) + 16
    profile = ProviderProfile("tight", max_bytes=budget, max_images=2)
    registry = ReferenceRegistry(refs)
    result = registry.pack([ref.ref_id for ref in refs], profile, required_ids=[ref.ref_id for ref in refs])
    assert len(result.refs) == 2
    assert not result.required_missing
    provider_inputs(result.prompt, [ref.data for ref in result.refs], profile)


@pytest.mark.parametrize("source_type", ("web", "avatar", "catalog"))
def test_inspected_candidate_survives_full_plus_four_set(source_type):
    registry = ReferenceRegistry(make_reference(1, number, image_bytes((number, 2, 3)), source_type=source_type)
                                 for number in range(8))
    inspected = registry.add(make_reference(1, 9, image_bytes("yellow")))
    working = registry.working_set(4, inspected_ids=(inspected.ref_id,))
    assert len(working.refs) == 4
    assert working.refs[0].ref_id == inspected.ref_id
    assert len(registry.refs) == 9


def test_prompt_budget_is_not_misreported_as_usable_text_fallback():
    registry = ReferenceRegistry()
    result = registry.pack((), ProviderProfile("tiny", max_bytes=64), prompt="x" * 65)
    assert result.status == "unavailable"
    assert result.issues[0].reason == "prompt_budget"
    assert registry.pack((), ProviderProfile("tiny", max_bytes=1)).status == "unavailable"


def test_invalid_provider_limits_are_still_rejected():
    registry = ReferenceRegistry()
    with pytest.raises(ValueError, match="max_images"):
        registry.pack((), ProviderProfile("bad", max_images=True))
    with pytest.raises(ValueError, match="max_refs"):
        registry.working_set(-1)


def test_packed_digest_matches_bytes_but_origin_digest_is_preserved():
    from hashlib import sha256

    ref = make_reference(1, 1, image_bytes(size=(1500, 1500)))
    result = ReferenceRegistry((ref,)).pack((ref.ref_id,), ProviderProfile("tight", max_bytes=4000))
    assert result.refs[0].digest == sha256(result.refs[0].data).hexdigest()
    assert result.refs[0].provenance[0].digest == ref.digest


def test_protected_set_reports_inspected_overflow_instead_of_losing_it():
    user = make_reference(0, 1, image_bytes(), source_type="submitted")
    inspected = make_reference(1, 1, image_bytes("blue"))
    registry = ReferenceRegistry((user, inspected))
    working = registry.working_set(1, inspected_ids=(inspected.ref_id,))
    assert working.refs == (user,)
    assert working.overflow_ids == (inspected.ref_id,)
    assert registry.resolve(inspected.ref_id) == inspected
