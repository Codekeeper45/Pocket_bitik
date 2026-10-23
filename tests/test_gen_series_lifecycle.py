import asyncio
import json

import pytest

import gen_series


def single_page_plan():
    return gen_series.parse_multipage_plan('PAGE 1: A fox', 1)


def test_delivery_phases_are_persisted_before_each_side_effect():
    plan = single_page_plan()
    snapshots, sends = [], []

    def save(state):
        snapshots.append(state.to_json())

    def deliver(page, result):
        assert json.loads(snapshots[-1])["delivery_phases"] == {"1": "sending"}
        sends.append(result)

    state = asyncio.run(gen_series.execute_series(plan, generate=lambda **kwargs: b"image",
                                                   save_checkpoint=save, deliver=deliver))
    assert [json.loads(raw)["delivery_phases"]["1"] for raw in snapshots] == ["not_started", "sending", "delivered"]
    assert sends == [b"image"]
    assert state.delivered == {1}
    assert state.delivery_phase(1) == "delivered"


def test_crash_after_artifact_save_before_first_send_resumes_without_regeneration():
    plan = single_page_plan()
    saved, generated, delivered = [], [], []

    def generate(**kwargs):
        generated.append(1)
        return {"artifact": "saved-image.png"}

    def crash_after_save(state):
        saved.append(state.to_json())
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(gen_series.execute_series(plan, generate=generate, save_checkpoint=crash_after_save,
                                               deliver=lambda *args: delivered.append(1)))
    state = gen_series.SeriesCheckpoint.from_json(saved[-1], plan=plan)
    assert state.delivery_phase(1) == "not_started"
    resumed = asyncio.run(gen_series.execute_series(plan, checkpoint=state, generate=generate,
                                                     deliver=lambda page, artifact: delivered.append(artifact)))
    assert generated == [1]
    assert delivered == [{"artifact": "saved-image.png"}]
    assert resumed.delivered == {1}


@pytest.mark.parametrize("phase", ["sending", "unknown"])
@pytest.mark.parametrize("reconciliation", [None, True, False])
def test_uncertain_and_legacy_pages_only_send_after_confirmed_absence(phase, reconciliation):
    plan = single_page_plan()
    state = gen_series.SeriesCheckpoint(plan.plan_id, completed={1: b"image"})
    if phase != "unknown":
        state.delivery_phases[1] = phase
    saved = json.loads(state.to_json())
    if phase == "unknown":
        saved.pop("delivery_phases")
    state = gen_series.SeriesCheckpoint.from_json(json.dumps(saved), plan=plan)
    assert state.delivery_phase(1) == phase
    sends, checks = [], []

    async def reconcile(page, artifact):
        checks.append((page.number, artifact))
        return reconciliation

    resumed = asyncio.run(gen_series.execute_series(
        plan, checkpoint=state, generate=lambda **kwargs: pytest.fail("regenerated saved artifact"),
        deliver=lambda *args: sends.append(1), reconcile_delivery=reconcile))
    assert checks == [(1, b"image")]
    assert sends == ([1] if reconciliation is False else [])
    assert resumed.delivered == ({1} if reconciliation is not None else set())
    assert bool(resumed.failures) == (reconciliation is None)


def test_uncertain_page_without_reconciler_never_sends():
    plan = single_page_plan()
    state = gen_series.SeriesCheckpoint(plan.plan_id, completed={1: b"image"})
    resumed = asyncio.run(gen_series.execute_series(
        plan, checkpoint=state, generate=lambda **kwargs: pytest.fail("regenerated"),
        deliver=lambda *args: pytest.fail("blind resend")))
    assert "reconciliation" in resumed.failures[1]
    assert resumed.delivery_phase(1) == "unknown"


def test_cancellation_during_send_leaves_sending_checkpoint():
    plan = single_page_plan()
    snapshots = []

    async def deliver(*args):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(gen_series.execute_series(plan, generate=lambda **kwargs: b"image", deliver=deliver,
                                               save_checkpoint=lambda state: snapshots.append(state.to_json())))
    state = gen_series.SeriesCheckpoint.from_json(snapshots[-1], plan=plan)
    assert state.delivery_phase(1) == "sending"
    assert state.delivered == set()


def test_failed_sending_checkpoint_prevents_send():
    plan = single_page_plan()

    def save(state):
        if state.delivery_phase(1) == "sending":
            raise OSError("disk unavailable")

    state = asyncio.run(gen_series.execute_series(plan, generate=lambda **kwargs: b"image", save_checkpoint=save,
                                                   deliver=lambda *args: pytest.fail("sent without durable sending marker")))
    assert state.completed == {1: b"image"}
    assert state.failures and not state.delivered


def test_post_send_checkpoint_failure_never_repeats_send():
    plan = single_page_plan()
    sends, snapshots = [], []

    def save(state):
        if state.delivered:
            raise OSError("disk unavailable after send")
        snapshots.append(state.to_json())

    state = asyncio.run(gen_series.execute_series(
        plan, generate=lambda **kwargs: b"image", save_checkpoint=save,
        deliver=lambda *args: sends.append(1)))
    assert state.delivered == {1} and state.failures
    assert gen_series.SeriesCheckpoint.from_json(snapshots[-1], plan=plan).delivery_phase(1) == "sending"
    asyncio.run(gen_series.execute_series(plan, checkpoint=state, generate=lambda **kwargs: pytest.fail("regenerated"),
                                          deliver=lambda *args: sends.append(1)))
    assert sends == [1]


def test_retrying_post_send_persistence_failure_does_not_claim_completion():
    plan = single_page_plan()
    state = gen_series.SeriesCheckpoint(plan.plan_id, completed={1: b"image"}, delivered={1},
                                        failures={1: "checkpoint OSError"})

    def save(state):
        raise OSError("still unavailable")

    resumed = asyncio.run(gen_series.execute_series(
        plan, checkpoint=state, generate=lambda **kwargs: pytest.fail("regenerated"),
        deliver=lambda *args: pytest.fail("resent"), save_checkpoint=save))
    assert resumed.failures == {1: "checkpoint OSError"}
    assert resumed.delivered == {1}


@pytest.mark.parametrize("phases,delivered", [
    ({"1": "invalid"}, []), ({"1": "delivered"}, []),
    ({"1": "not_started"}, [1]), ({"2": "sending"}, []),
    ({"01": "sending"}, []), ([], []),
])
def test_invalid_delivery_phases_are_rejected(phases, delivered):
    plan = single_page_plan()
    payload = json.loads(gen_series.SeriesCheckpoint(plan.plan_id, completed={1: b"image"}).to_json())
    payload.update(delivery_phases=phases, delivered=delivered)
    with pytest.raises(gen_series.SeriesPlanError):
        gen_series.SeriesCheckpoint.from_json(json.dumps(payload), plan=plan)


def test_legacy_delivered_markers_and_reconciled_markers_round_trip():
    plan = single_page_plan()
    state = gen_series.SeriesCheckpoint(plan.plan_id, completed={1: b"image"}, delivered={1})
    payload = json.loads(state.to_json())
    payload.pop("delivery_phases")
    restored = gen_series.SeriesCheckpoint.from_json(json.dumps(payload), plan=plan)
    assert restored.delivery_phase(1) == "delivered"
    restored.delivery_phases[1] = "sending"
    assert gen_series.SeriesCheckpoint.from_json(restored.to_json(), plan=plan).delivery_phase(1) == "delivered"


def test_shared_job_lock_serializes_load_and_execution():
    async def scenario():
        plan = single_page_plan()
        saved, loads, generated, sent = [], [], [], []
        started, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            generated.append(1)
            started.set()
            await release.wait()
            return b"image"

        async def run():
            async with gen_series.series_job_lock("same-job"):
                loads.append(1)
                checkpoint = gen_series.SeriesCheckpoint.from_json(saved[-1], plan=plan) if saved else None
                return await gen_series.execute_series(
                    plan, generate=generate, checkpoint=checkpoint,
                    save_checkpoint=lambda state: saved.append(state.to_json()),
                    deliver=lambda *args: sent.append(1))

        first = asyncio.create_task(run())
        await started.wait()
        second = asyncio.create_task(run())
        await asyncio.sleep(0)
        assert loads == [1]
        release.set()
        await asyncio.gather(first, second)
        assert loads == [1, 1] and generated == sent == [1]

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


def test_job_lock_cancellation_and_independent_jobs():
    async def scenario():
        async def acquire(job):
            async with gen_series.series_job_lock(job):
                return job

        async with gen_series.series_job_lock("first"):
            waiter = asyncio.create_task(acquire("first"))
            await asyncio.sleep(0)
            assert not waiter.done()
            assert await acquire("second") == "second"
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert await acquire("first") == "first"

    asyncio.run(asyncio.wait_for(scenario(), timeout=2))


@pytest.mark.parametrize("job_id", ["", "../escape", "bad/id"])
def test_job_lock_rejects_invalid_ids(job_id):
    async def scenario():
        async with gen_series.series_job_lock(job_id):
            pytest.fail("invalid job id accepted")

    with pytest.raises(gen_series.SeriesPlanError):
        asyncio.run(scenario())


@pytest.mark.parametrize("count", [1, 3])
def test_raw_plan_round_trip_preserves_exact_prompt_and_checkpoint_identity(count):
    from gen_prompt import parse_intent_plan
    source = '  Keep image #1 and literal "Привет!"\n'
    intent = parse_intent_plan(None, source, raw=True, explicit_count=count)
    plan = gen_series.series_plan_from_intent(intent)
    saved = json.dumps({"pages": [page.__dict__ for page in plan.pages],
                        "bible": plan.bible, "version": plan.version})
    restored = gen_series.parse_multipage_plan(saved, count)
    assert restored.plan_id == plan.plan_id
    assert all(page.prompt == source for page in restored.pages)
    checkpoint = gen_series.SeriesCheckpoint(plan.plan_id, completed={1: b"image"}, delivered={1})
    assert gen_series.SeriesCheckpoint.from_json(checkpoint.to_json(), plan=restored).delivered == {1}


@pytest.mark.parametrize("version,bible", [
    ("1", {}), ("1", {"raw": True, "output_kind": "independent_images"}),
    ("raw-1", {}), ("raw-1", {"raw": False, "output_kind": "independent_images"}),
    ("raw-1", {"raw": True, "output_kind": "series"}),
    ("raw-1", {"raw": True, "output_kind": "single_image"}),
])
def test_duplicate_prompt_exception_requires_explicit_independent_raw_metadata(version, bible):
    payload = {"pages": [{"number": 1, "prompt": "same"}, {"number": 2, "prompt": "same"}],
               "bible": bible, "version": version}
    with pytest.raises(gen_series.SeriesPlanError):
        gen_series.parse_multipage_plan(json.dumps(payload), 2)


def test_raw_metadata_does_not_allow_duplicate_page_numbers():
    payload = {"pages": [{"number": 1, "prompt": "same"}, {"number": 1, "prompt": "same"}],
               "bible": {"raw": True, "output_kind": "independent_images"}, "version": "raw-1"}
    with pytest.raises(gen_series.SeriesPlanError, match="duplicate page 1"):
        gen_series.parse_multipage_plan(json.dumps(payload), 2)
