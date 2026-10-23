from io import BytesIO

from PIL import Image
import pytest

from gen_references import Reference, ReferenceRegistry, make_reference


def image_bytes(color="red", size=(24, 24)):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, "PNG")
    return stream.getvalue()


def test_namespaces_and_content_versions_do_not_collide():
    submitted = make_reference(0, 1, image_bytes(), source_type="submitted")
    chat = make_reference(0, 1, image_bytes("blue"))
    web = make_reference(0, 1, image_bytes("green"), source_type="web")
    registry = ReferenceRegistry((submitted, chat, web))
    assert [ref.ref_id.split(":")[0] for ref in registry.refs] == ["submitted", "chat", "web"]
    registry.add(make_reference(0, 1, image_bytes("yellow")))
    assert len(registry.refs) == 4


def test_duplicate_bytes_merge_roles_provenance_and_aliases():
    first = make_reference(1, 1, image_bytes(), roles=("subject", "clothing"))
    second = make_reference(0, 1, image_bytes(), role="style", source_type="web",
                            image_url="https://example.org/image.png",
                            source_page_url="https://example.org/page")
    registry = ReferenceRegistry((first, second))
    assert len(registry.refs) == 1
    merged = registry.refs[0]
    assert merged.roles == ("subject", "clothing", "style")
    assert len(merged.provenance) == 2
    assert merged.provenance[1].source_page_url.endswith("/page")
    selected = registry.selected((second.ref_id, first.ref_id))
    assert len(selected) == 1 and selected[0].api_index == 1
    assert registry.refs[0].api_index is None
    assert registry.api_mapping(selected)[second.ref_id] == 1


def test_legacy_reference_and_optional_missing():
    legacy = Reference("old-checkpoint-id", 1, 2, "unused", image_bytes(), role="identity")
    registry = ReferenceRegistry((legacy,))
    assert registry.refs[0].roles == ("subject",)
    assert registry.selected(("old-checkpoint-id",))[0].ref_id == "old-checkpoint-id"
    assert registry.selected(("missing",), optional_ids=("missing",)) == ()


def test_structured_manifest_preserves_roles_without_rewriting_text():
    ref = make_reference(1, 1, image_bytes(), roles=("subject", "clothing"))
    registry = ReferenceRegistry((ref,))
    manifest = registry.manifest(registry.selected((ref.ref_id,)))
    assert manifest[0]["roles"] == ["subject", "clothing"]
    assert manifest[0]["api_index"] == 1
    assert "data" not in manifest[0]


def test_checkpoint_roundtrip_normalizes_provenance_lists_and_retains_aliases():
    from dataclasses import asdict
    from hashlib import sha256

    first = make_reference(1, 1, image_bytes(), roles=("subject", "clothing"))
    payload = asdict(first)
    payload["roles"] = list(payload["roles"])
    payload["provenance"][0]["roles"] = list(payload["provenance"][0]["roles"])
    restored = Reference(**payload)
    registry = ReferenceRegistry((restored,))
    duplicate = make_reference(2, 2, first.data, role="style")
    registry.add(duplicate)
    legacy_id = sha256(f"1:1:{first.digest}".encode()).hexdigest()[:24]
    assert registry.resolve(legacy_id).roles == ("subject", "clothing", "style")
    assert registry.resolve(duplicate.ref_id).ref_id == first.ref_id


def test_ambiguous_alias_is_rejected_without_mutating_registry():
    first = make_reference(1, 1, image_bytes(), aliases=("legacy:1",))
    registry = ReferenceRegistry((first,))
    other = make_reference(1, 2, image_bytes("blue"), aliases=("legacy:1",))
    with pytest.raises(ValueError, match="ID reused"):
        registry.add(other)
    assert registry.refs == (first,)


def test_submitted_source_is_prioritized_even_after_content_dedup():
    candidate = make_reference(1, 1, image_bytes(), priority=100)
    chat = make_reference(1, 2, image_bytes("blue"), role="style")
    submitted = make_reference(0, 1, chat.data, source_type="submitted", role="subject")
    registry = ReferenceRegistry((candidate, chat, submitted))
    chosen = registry.selected((candidate.ref_id, submitted.ref_id))
    assert chosen[0].ref_id == chat.ref_id
    assert chosen[0].roles == ("style", "subject")
    assert len(chosen[0].provenance) == 2


def test_repeated_roles_for_same_source_do_not_duplicate_origin_rows():
    first = make_reference(1, 1, image_bytes(), roles=("subject",))
    second = make_reference(1, 1, first.data, roles=("clothing",))
    registry = ReferenceRegistry((first, second))
    assert registry.refs[0].roles == ("subject", "clothing")
    assert len(registry.refs[0].provenance) == 1
    assert registry.refs[0].provenance[0].roles == ("subject", "clothing")
    assert first.role == "subject"
