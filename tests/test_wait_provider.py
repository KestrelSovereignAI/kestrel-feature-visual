"""Tests for the LoRA training Waitable provider and shared finalization.

No real RunPod / DB: a fake training provider records cleanup/cancel calls and
returns canned status snapshots; a fake db_pool/connection records the
avatar_config UPSERT. The point under test is finalize-once: the blocking loop
and the provider must funnel terminal side effects through a single idempotent
``_finalize_training``.
"""

import pytest

from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_feature_visual.wait_provider import LoraTrainingWaitable


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeStatus:
    """Mimics TrainingStatus enough for poll() / _finalize_training()."""

    def __init__(self, state, *, progress=0.5, error=None, provider_details=None):
        self.state = state
        self.progress = progress
        self.error = error
        self.provider_details = provider_details or {}


class FakeTrainingProvider:
    """Records cleanup/cancel; returns a scripted sequence of statuses."""

    provider_name = "fake_runpod"

    def __init__(self, statuses):
        # statuses: list of FakeStatus returned on successive get_status calls;
        # the last is repeated once exhausted.
        self._statuses = list(statuses)
        self.cleanup_calls = []
        self.cancel_calls = []
        self.cleanup_raises = False

    async def get_status(self, job_id):
        if len(self._statuses) > 1:
            return self._statuses.pop(0)
        return self._statuses[0]

    async def cleanup(self, job_id):
        self.cleanup_calls.append(job_id)
        if self.cleanup_raises:
            raise RuntimeError("session already torn down")

    async def cancel(self, job_id):
        self.cancel_calls.append(job_id)


class FakeConn:
    """Fake asyncpg-style connection over a shared in-memory companion store.

    ``rows`` maps companion_id -> avatar_config dict, mutated by the same jsonb
    ``||`` merge semantics the real ``UPDATE`` uses, so finalize/dispatch writes
    are observable by later ``fetchrow``/``fetch`` calls (active_handles,
    trigger-word recovery).
    """

    def __init__(self, rows=None):
        self.executed = []
        # companion_id -> avatar_config dict
        self.rows = rows if rows is not None else {}

    async def execute(self, sql, *params):
        self.executed.append((sql, params))
        # Emulate: UPDATE companions SET avatar_config = COALESCE(...) || $1 WHERE id = $2
        if "avatar_config" in sql and "||" in sql and len(params) == 2:
            import json as _json

            merge = _json.loads(params[0])
            companion_id = params[1]
            current = self.rows.get(companion_id) or {}
            if not isinstance(current, dict):
                current = {}
            current.update(merge)
            self.rows[companion_id] = current

    async def fetchrow(self, sql, *params):
        companion_id = params[0]
        if companion_id not in self.rows:
            return None
        return {"avatar_config": self.rows.get(companion_id)}

    async def fetch(self, sql, *params):
        # SELECT id, avatar_config ... WHERE lora_training_status IN
        # ('running', 'finalizing') — a 'finalizing' job (terminal metadata
        # persisted, pod teardown not yet confirmed) stays enumerated so the
        # reconciler keeps re-polling and cleanup is retried.
        out = []
        for cid, cfg in self.rows.items():
            cfg = cfg if isinstance(cfg, dict) else {}
            if cfg.get("lora_training_status") in ("running", "finalizing"):
                out.append({"id": cid, "avatar_config": cfg})
        return out


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeDbPool:
    def __init__(self, rows=None):
        self.conn = FakeConn(rows)

    def acquire(self):
        return _Acquire(self.conn)


def _training_state():
    from kestrel_sovereign.features.training.types import TrainingState

    return TrainingState


def make_feature(provider, db_pool=None):
    """Build a real VisualIdentityFeature wired to fakes (no initialize()).

    Installs a STRICT ``_get_recorded_provider`` shim that resolves a recorded
    provider name to ``provider`` ONLY when the name matches
    ``provider.provider_name`` (mirroring the production strict-resolution
    contract: never fall back to a different backend). The real
    ``_get_recorded_provider`` goes through ``TrainingProviderFactory``, which
    has no knowledge of the test fakes, so tests wire it here.
    """
    from kestrel_feature_visual.feature import VisualIdentityFeature

    feature = VisualIdentityFeature(agent=None)
    feature._training_provider = provider
    feature.db_pool = db_pool
    feature._finalized_jobs = set()
    feature._cleanup_attempts = {}

    def strict_recorded(provider_name=None):
        if not provider_name:
            return None
        if provider is not None and getattr(provider, "provider_name", None) == provider_name:
            return provider
        return None

    feature._get_recorded_provider = strict_recorded
    return feature


# ---------------------------------------------------------------------------
# Handle parsing
# ---------------------------------------------------------------------------

class TestParse:
    def test_simple_handle(self):
        assert LoraTrainingWaitable._parse("comp-1:job-9") == ("comp-1", "job-9")

    def test_job_id_with_colons(self):
        # job_id may itself contain colons; only the first split counts.
        assert LoraTrainingWaitable._parse("comp-1:runpod:abc:123") == (
            "comp-1",
            "runpod:abc:123",
        )

    def test_no_colon_is_jobid_only(self):
        assert LoraTrainingWaitable._parse("loose-job") == ("", "loose-job")


# ---------------------------------------------------------------------------
# Poll classification
# ---------------------------------------------------------------------------

class TestPollClassification:
    @pytest.mark.asyncio
    async def test_running_is_pending(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.TRAINING, progress=0.4)])
        feature = make_feature(provider, FakeDbPool())
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.PENDING
        assert status.data["state"] == "training"
        assert provider.cleanup_calls == []

    @pytest.mark.asyncio
    async def test_completed_is_done(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        db = FakeDbPool()
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.DONE
        assert status.data["lora_path"] == "fake_runpod:job-9"
        # finalize ran: cleanup + persistence. Two writes now: the intermediate
        # "finalizing" (metadata persisted) then "completed" (committed after
        # cleanup succeeds) — codex P2-B.
        assert provider.cleanup_calls == ["job-9"]
        assert len(db.conn.executed) == 2

    @pytest.mark.asyncio
    async def test_failed_is_failed(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.FAILED, error="boom")])
        feature = make_feature(provider, FakeDbPool())
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.FAILED
        assert "boom" in status.summary
        assert provider.cleanup_calls == ["job-9"]

    @pytest.mark.asyncio
    async def test_cancelled_is_failed(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.CANCELLED)])
        feature = make_feature(provider, FakeDbPool())
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.FAILED
        assert provider.cleanup_calls == ["job-9"]

    @pytest.mark.asyncio
    async def test_malformed_handle_rejected_without_finalizing(self):
        """A handle missing the companion prefix must be rejected BEFORE any
        finalize so it can't poison the _finalized_jobs guard (codex P2)."""
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        db = FakeDbPool()
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        status = await w.poll("job-9")  # no companion prefix -> companion_id=""

        assert status.outcome is Outcome.FAILED
        assert "malformed" in status.summary
        # No finalization side effects, no poisoned guard.
        assert provider.cleanup_calls == []
        assert "job-9" not in feature._finalized_jobs
        assert db.conn.executed == []

    @pytest.mark.asyncio
    async def test_no_provider_is_failed(self):
        feature = make_feature(None, FakeDbPool())
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.FAILED
        assert "provider unavailable" in status.summary

    @pytest.mark.asyncio
    async def test_poll_lazily_reinitializes_provider_after_restart(self):
        """After a restart _training_provider is None even though active_handles
        can discover persisted running jobs; poll must rebuild the provider via
        the lazy getter rather than reporting FAILED (codex P1)."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        rebuilt = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        rebuilt.provider_name = "vertex_ai"  # rebuilt as the RECORDED backend
        feature = make_feature(None, db)  # simulate post-restart: no cached provider

        def strict_rebuild(provider_name=None):
            if provider_name == "vertex_ai":
                feature._training_provider = rebuilt
                return rebuilt
            return None

        feature._get_recorded_provider = strict_rebuild
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.DONE
        assert rebuilt.cleanup_calls == ["job-9"]
        assert db.conn.rows["comp-1"]["lora_training_status"] == "completed"


# ---------------------------------------------------------------------------
# Poll's terminal path invokes the shared finalizer (exactly once)
# ---------------------------------------------------------------------------

class TestPollInvokesFinalize:
    @pytest.mark.asyncio
    async def test_terminal_poll_calls_finalize_once(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        feature = make_feature(provider, FakeDbPool())

        calls = []
        orig = feature._finalize_training

        async def spy(**kwargs):
            calls.append(kwargs)
            return await orig(**kwargs)

        feature._finalize_training = spy
        w = LoraTrainingWaitable(feature)

        await w.poll("comp-1:job-9")

        assert len(calls) == 1
        assert calls[0]["companion_id"] == "comp-1"
        assert calls[0]["job_id"] == "job-9"
        assert calls[0]["terminal_state"] == TS.COMPLETED


# ---------------------------------------------------------------------------
# Idempotency: two terminal polls -> cleanup + persist once
# ---------------------------------------------------------------------------

class TestIdempotency:
    @pytest.mark.asyncio
    async def test_two_terminal_polls_cleanup_once(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        db = FakeDbPool()
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        first = await w.poll("comp-1:job-9")
        second = await w.poll("comp-1:job-9")

        # Both observations report DONE...
        assert first.outcome is Outcome.DONE
        assert second.outcome is Outcome.DONE
        assert second.data["lora_path"] == "fake_runpod:job-9"
        # ...but the side effects ran exactly once: cleanup once, and the
        # two-phase write (finalizing + completed) happened on the FIRST poll
        # only; the second poll hits the _finalized_jobs guard and writes nothing.
        assert provider.cleanup_calls == ["job-9"]
        assert len(db.conn.executed) == 2

    @pytest.mark.asyncio
    async def test_finalize_direct_double_call_runs_once(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        db = FakeDbPool()
        feature = make_feature(provider, db)

        p1 = await feature._finalize_training(
            companion_id="comp-1", job_id="job-9",
            terminal_state=TS.COMPLETED, provider="fake_runpod",
            trigger_word="TOK", output_path=None,
        )
        p2 = await feature._finalize_training(
            companion_id="comp-1", job_id="job-9",
            terminal_state=TS.COMPLETED, provider="fake_runpod",
            trigger_word="TOK", output_path=None,
        )

        assert p1 == "fake_runpod:job-9"
        assert p2 == "fake_runpod:job-9"
        assert provider.cleanup_calls == ["job-9"]
        # First call: two-phase write (finalizing + completed). Second call hits
        # the _finalized_jobs guard and writes nothing.
        assert len(db.conn.executed) == 2

    @pytest.mark.asyncio
    async def test_cleanup_targets_recorded_provider_not_default(self):
        """codex P1: finalize must tear down the provider the job RAN on, not
        the cached default — else a restart / non-default backend leaves the
        real pod billing."""
        TS = _training_state()
        default_provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        recorded_provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        db = FakeDbPool()
        feature = make_feature(default_provider, db)

        # Strictly resolve "vertex_ai" to the recorded provider; anything else
        # (incl. the default's name) resolves to None — the strict resolver
        # NEVER falls back to the cached default for a recorded job.
        def resolve(provider_name=None):
            if provider_name == "vertex_ai":
                return recorded_provider
            return None
        feature._get_recorded_provider = resolve

        await feature._finalize_training(
            companion_id="comp-1", job_id="job-9",
            terminal_state=TS.COMPLETED, provider="vertex_ai",
            trigger_word="TOK", output_path=None,
        )

        # The job's actual backend was cleaned up; the cached default was NOT.
        assert recorded_provider.cleanup_calls == ["job-9"]
        assert default_provider.cleanup_calls == []

    @pytest.mark.asyncio
    async def test_concurrent_terminal_observers_run_side_effects_once(self):
        """Two coroutines observing the same job terminal AT ONCE must still run
        persistence + cleanup exactly once — the per-job lock makes the
        check-then-add guard atomic across the awaited DB write (codex P2)."""
        import asyncio

        TS = _training_state()

        # A db whose execute() awaits a real event loop hop, widening the window
        # in which both coroutines could otherwise pass the membership check.
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        original_execute = db.conn.execute

        async def slow_execute(sql, *params):
            await asyncio.sleep(0)  # force a scheduler yield mid-finalize
            return await original_execute(sql, *params)

        db.conn.execute = slow_execute
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        feature = make_feature(provider, db)
        # The job's recorded provider is "vertex_ai"; resolve it strictly to the
        # fake so cleanup targets the right (only) backend.
        feature._get_recorded_provider = lambda provider_name=None: (
            provider if provider_name == "vertex_ai" else None
        )

        r1, r2 = await asyncio.gather(
            feature._finalize_training(
                companion_id="comp-1", job_id="job-9",
                terminal_state=TS.COMPLETED, provider="vertex_ai",
                trigger_word="TOK", output_path=None,
            ),
            feature._finalize_training(
                companion_id="comp-1", job_id="job-9",
                terminal_state=TS.COMPLETED, provider="vertex_ai",
                trigger_word="TOK", output_path=None,
            ),
        )

        assert r1 == r2 == "vertex_ai:job-9"
        # Side effects ran exactly once despite concurrent observation.
        assert provider.cleanup_calls == ["job-9"]
        completed_writes = [
            e for e in db.conn.executed if "lora_model_path" in str(e[1])
        ]
        assert len(completed_writes) == 1

    @pytest.mark.asyncio
    async def test_cleanup_exception_is_tolerated(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        provider.cleanup_raises = True
        feature = make_feature(provider, FakeDbPool())

        # Must not raise even though provider.cleanup blows up.
        path = await feature._finalize_training(
            companion_id="comp-1", job_id="job-9",
            terminal_state=TS.COMPLETED, provider="fake_runpod",
            trigger_word=None, output_path=None,
        )
        assert path == "fake_runpod:job-9"
        assert provider.cleanup_calls == ["job-9"]


# ---------------------------------------------------------------------------
# Dispatch records canonical in-flight metadata
# ---------------------------------------------------------------------------

class TestDispatchRecordsMetadata:
    @pytest.mark.asyncio
    async def test_record_inflight_writes_running_metadata(self):
        db = FakeDbPool()
        feature = make_feature(FakeTrainingProvider([FakeStatus(_training_state().TRAINING)]), db)

        await feature._record_inflight_training(
            companion_id="comp-1",
            job_id="job-9",
            trigger_word="TOKalice",
            provider="vertex_ai",
            output_path=None,
        )

        cfg = db.conn.rows["comp-1"]
        assert cfg["lora_training_status"] == "running"
        assert cfg["lora_job_id"] == "job-9"
        assert cfg["lora_trigger_word"] == "TOKalice"
        assert cfg["lora_provider"] == "vertex_ai"
        # output_path was None -> not persisted (no clobber).
        assert "lora_output_path" not in cfg

    @pytest.mark.asyncio
    async def test_record_inflight_omits_none_fields(self):
        db = FakeDbPool()
        feature = make_feature(FakeTrainingProvider([FakeStatus(_training_state().TRAINING)]), db)

        await feature._record_inflight_training(
            companion_id="comp-1", job_id="job-9",
            trigger_word=None, provider=None, output_path="gs://b/out",
        )

        cfg = db.conn.rows["comp-1"]
        assert "lora_trigger_word" not in cfg
        assert "lora_provider" not in cfg
        assert cfg["lora_output_path"] == "gs://b/out"
        assert cfg["lora_training_status"] == "running"


# ---------------------------------------------------------------------------
# Finalize recovers canonical metadata; never clobbers with null
# ---------------------------------------------------------------------------

class TestFinalizeMetadataCorrectness:
    @pytest.mark.asyncio
    async def test_trigger_word_recovered_not_nulled(self):
        """Provider path passes trigger_word=None; finalize must recover the
        dispatch-recorded trigger_word and NOT overwrite it with null."""
        TS = _training_state()
        # Pre-seed the dispatch-recorded running record.
        db = FakeDbPool(rows={
            "comp-1": {
                "lora_training_status": "running",
                "lora_job_id": "job-9",
                "lora_trigger_word": "TOKalice",
                "lora_provider": "vertex_ai",
            }
        })
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        provider.provider_name = "vertex_ai"  # match the recorded backend (strict resolve)
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        # finalize keeps provider from the status path (non-None), trigger_word
        # recovered from config.
        await w.poll("comp-1:job-9")

        cfg = db.conn.rows["comp-1"]
        assert cfg["lora_trigger_word"] == "TOKalice"  # NOT nulled
        assert cfg["lora_training_status"] == "completed"

    @pytest.mark.asyncio
    async def test_gcs_output_path_resolved_from_provider_details(self):
        """Vertex AI exposes the real path at provider_details['gcs_output_path'];
        finalize must persist THAT, not the opaque '<provider>:<job_id>'."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {
                "lora_training_status": "running",
                "lora_job_id": "job-9",
                "lora_trigger_word": "TOKalice",
                "lora_provider": "vertex_ai",
            }
        })
        provider = FakeTrainingProvider([
            FakeStatus(TS.COMPLETED, provider_details={
                "gcs_output_path": "gs://bucket/loras/comp-1/weights.safetensors"
            })
        ])
        provider.provider_name = "vertex_ai"  # match the recorded backend (strict resolve)
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.data["lora_path"] == "gs://bucket/loras/comp-1/weights.safetensors"
        cfg = db.conn.rows["comp-1"]
        assert cfg["lora_model_path"] == "gs://bucket/loras/comp-1/weights.safetensors"
        assert "vertex_ai:job-9" != cfg["lora_model_path"]

    def test_resolve_lora_path_prefers_gcs_then_output_then_fallback(self):
        from kestrel_feature_visual.feature import VisualIdentityFeature

        rp = VisualIdentityFeature._resolve_lora_path
        # gcs_output_path wins
        assert rp("vertex_ai", "j", "gs://top", {"gcs_output_path": "gs://gcs"}) == "gs://gcs"
        # provider_details.output_path next
        assert rp("vertex_ai", "j", None, {"output_path": "gs://pd"}) == "gs://pd"
        # top-level output_path next
        assert rp("vertex_ai", "j", "gs://top", {}) == "gs://top"
        # fallback sentinel
        assert rp("vertex_ai", "j", None, {}) == "vertex_ai:j"

    @pytest.mark.asyncio
    async def test_db_persist_failure_leaves_job_retryable(self):
        """A transient DB failure during terminal finalization must NOT mark the
        job finalized — the row stays 'running' so a later poll/reconciler
        retries and the LoRA is eventually persisted (codex P2)."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })

        # First execute() (the terminal UPDATE) raises; subsequent ones succeed.
        original_execute = db.conn.execute
        state = {"fail_next": True}

        async def flaky_execute(sql, *params):
            if state["fail_next"] and "lora_model_path" in str(params):
                state["fail_next"] = False
                raise RuntimeError("transient db error")
            return await original_execute(sql, *params)

        db.conn.execute = flaky_execute
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        provider.provider_name = "vertex_ai"  # match the recorded backend (strict resolve)
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        # First poll: DB write fails -> job NOT finalized, still 'running', and
        # the wait must NOT report DONE (the LoRA path isn't persisted yet).
        first = await w.poll("comp-1:job-9")
        assert first.outcome is Outcome.PENDING
        assert first.data.get("finalization_pending") is True
        assert "job-9" not in feature._finalized_jobs
        assert db.conn.rows["comp-1"]["lora_training_status"] == "running"
        assert await w.active_handles() == ["comp-1:job-9"]

        # Second poll (reconciler retry): DB write succeeds -> persisted + done.
        status = await w.poll("comp-1:job-9")
        assert status.outcome is Outcome.DONE
        assert "job-9" in feature._finalized_jobs
        assert db.conn.rows["comp-1"]["lora_training_status"] == "completed"
        assert db.conn.rows["comp-1"]["lora_trigger_word"] == "TOK"
        assert await w.active_handles() == []

    @pytest.mark.asyncio
    async def test_failed_marks_status_failed(self):
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9"}
        })
        provider = FakeTrainingProvider([FakeStatus(TS.FAILED, error="boom")])
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        await w.poll("comp-1:job-9")

        assert db.conn.rows["comp-1"]["lora_training_status"] == "failed"


# ---------------------------------------------------------------------------
# active_handles enumerates in-flight jobs (MonitorableWaitable)
# ---------------------------------------------------------------------------

class TestActiveHandles:
    @pytest.mark.asyncio
    async def test_returns_running_jobs_as_handles(self):
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9"},
            "comp-2": {"lora_training_status": "running", "lora_job_id": "runpod:abc:123"},
            "comp-3": {"lora_training_status": "completed", "lora_job_id": "job-x"},
        })
        feature = make_feature(FakeTrainingProvider([FakeStatus(_training_state().TRAINING)]), db)
        w = LoraTrainingWaitable(feature)

        handles = await w.active_handles()

        assert set(handles) == {"comp-1:job-9", "comp-2:runpod:abc:123"}

    @pytest.mark.asyncio
    async def test_no_db_pool_returns_empty(self):
        feature = make_feature(FakeTrainingProvider([FakeStatus(_training_state().TRAINING)]), None)
        w = LoraTrainingWaitable(feature)

        assert await w.active_handles() == []

    @pytest.mark.asyncio
    async def test_completed_job_no_longer_enumerated(self):
        """After finalize marks a job completed, active_handles drops it."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        provider.provider_name = "vertex_ai"  # match the recorded backend (strict resolve)
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        assert await w.active_handles() == ["comp-1:job-9"]
        await w.poll("comp-1:job-9")
        assert await w.active_handles() == []

    @pytest.mark.asyncio
    async def test_poll_uses_recorded_provider_not_default(self):
        """A persisted job must be polled on the provider it was dispatched on
        (avatar_config.lora_provider), via the STRICT resolver — not the current
        default (codex P2)."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vastai"},
        })
        recorded = FakeTrainingProvider([FakeStatus(TS.TRAINING)])
        feature = make_feature(None, db)

        requested = []

        def strict_get(provider_name=None):
            requested.append(provider_name)
            if provider_name == "vastai":
                return recorded
            return None

        feature._get_recorded_provider = strict_get
        # The default getter must NOT be consulted for a recorded job; trip the
        # test loudly if poll ever falls back to it.
        def forbidden_default(provider_name=None):
            raise AssertionError("recorded job must not use the default getter")
        feature._get_training_provider = forbidden_default
        w = LoraTrainingWaitable(feature)

        await w.poll("comp-1:job-9")

        # The STRICT resolver was asked for the RECORDED provider, not the default.
        assert requested == ["vastai"]

    def test_provider_is_structurally_monitorable(self):
        """The class now satisfies MonitorableWaitable (poll + active_handles)."""
        from kestrel_sdk.tools.waitable import MonitorableWaitable

        feature = make_feature(FakeTrainingProvider([FakeStatus(_training_state().TRAINING)]), FakeDbPool())
        w = LoraTrainingWaitable(feature)
        assert isinstance(w, MonitorableWaitable)
        assert hasattr(w, "poll") and hasattr(w, "active_handles")


# ---------------------------------------------------------------------------
# TrainingState-unimportable defensive string fallback
# ---------------------------------------------------------------------------

class TestStateFallback:
    @pytest.mark.asyncio
    async def test_string_state_classified_without_enum(self, monkeypatch):
        """If TrainingState can't be imported, poll classifies via the raw
        ``.value`` string compare instead."""
        import kestrel_feature_visual.wait_provider as wp

        class StrState:
            value = "completed"

        provider = FakeTrainingProvider([FakeStatus(StrState())])
        db = FakeDbPool()
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        # Force the host enum import to fail inside the provider's classifier
        # and terminal-state builder.
        real_import = wp.LoraTrainingWaitable._classify

        def fail_classify(self, state, state_value):
            v = state_value.lower()
            return (v == "completed", v == "failed", v == "cancelled")

        # Stub finalize so we don't depend on the host enum inside the feature
        # (the point here is the PROVIDER's string-compare path).
        finalize_calls = []

        async def fake_finalize(**kwargs):
            finalize_calls.append(kwargs)
            # Mirror the real finalizer marking the job persisted so poll reports
            # DONE rather than the finalization-pending retry path.
            feature._finalized_jobs.add(kwargs["job_id"])
            return "fake_runpod:job-9"

        feature._finalize_training = fake_finalize
        monkeypatch.setattr(
            wp.LoraTrainingWaitable, "_classify", fail_classify, raising=True
        )

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.DONE
        assert len(finalize_calls) == 1
        # terminal_state passed through is the string fallback, not the enum,
        # because we forced classify to the string path. Builder still tries
        # the enum first; assert it is a recognizable completed marker.
        ts = finalize_calls[0]["terminal_state"]
        assert str(getattr(ts, "value", ts)) == "completed"


# ---------------------------------------------------------------------------
# codex P2-A: strict recorded-provider resolution — never poll/clean up the
# WRONG backend. A recorded job whose backend resolves to a DIFFERENT
# provider_name must NOT be polled or torn down against that backend; poll
# reports PENDING and no cleanup runs (so the real pod keeps being retried, not
# wrongly marked terminal while it bills).
# ---------------------------------------------------------------------------

class TestStrictRecordedProvider:
    @pytest.mark.asyncio
    async def test_poll_pending_when_recorded_provider_unavailable(self):
        """Recorded job's backend can't be resolved right now -> poll returns
        PENDING ('recorded provider ... unavailable'), no cleanup on any
        backend, status stays running, job still enumerated."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        # A default/wrong-backend provider that WOULD report COMPLETED + clean up
        # if (incorrectly) consulted. The strict resolver must never hand it back
        # for a recorded "vertex_ai" job.
        wrong = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        wrong.provider_name = "runpod"
        feature = make_feature(wrong, db)
        # Strict resolver cannot resolve "vertex_ai" (e.g. creds absent) -> None.
        feature._get_recorded_provider = lambda provider_name=None: None
        # If poll ever fell back to the default getter, it'd find the wrong pod.
        feature._get_training_provider = lambda provider_name=None: wrong
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.PENDING
        assert "recorded provider vertex_ai unavailable" in status.summary
        # No cleanup against the wrong backend, no terminal write.
        assert wrong.cleanup_calls == []
        assert wrong.get_status  # not consulted, but exists
        assert db.conn.rows["comp-1"]["lora_training_status"] == "running"
        assert "job-9" not in feature._finalized_jobs
        assert await w.active_handles() == ["comp-1:job-9"]

    @pytest.mark.asyncio
    async def test_safe_cleanup_refuses_wrong_backend(self):
        """_safe_cleanup must report failure (not clean up a different backend)
        when the recorded provider name resolves to None — so the finalizer
        leaves the job 'finalizing' and retries, never tearing down the wrong
        pod (codex P2-A)."""
        wrong = FakeTrainingProvider([])
        wrong.provider_name = "runpod"
        feature = make_feature(wrong, FakeDbPool())
        feature._get_recorded_provider = lambda provider_name=None: None

        ok = await feature._safe_cleanup("job-9", "vertex_ai")

        assert ok is False
        assert wrong.cleanup_calls == []


# ---------------------------------------------------------------------------
# codex P2-B: cleanup failure leaves the job 'finalizing' (still enumerated,
# NOT guarded), and a later finalize retries cleanup until it succeeds. Bounded
# by MAX_CLEANUP_ATTEMPTS so a permanently-erroring cleanup can't loop forever.
# ---------------------------------------------------------------------------

class TestFinalizingRetry:
    @pytest.mark.asyncio
    async def test_cleanup_failure_stays_finalizing_then_retry_succeeds(self):
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        provider.provider_name = "vertex_ai"
        # First cleanup attempt fails transiently; later attempts succeed.
        state = {"fail": True}

        async def flaky_cleanup(job_id):
            provider.cleanup_calls.append(job_id)
            if state["fail"]:
                state["fail"] = False
                raise RuntimeError("connection reset")
        provider.cleanup = flaky_cleanup
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        # First poll: terminal observed, metadata persisted as "finalizing",
        # cleanup FAILS -> stays finalizing, NOT in guard, still enumerated.
        first = await w.poll("comp-1:job-9")
        assert first.outcome is Outcome.PENDING  # finalization not yet committed
        assert db.conn.rows["comp-1"]["lora_training_status"] == "finalizing"
        assert db.conn.rows["comp-1"]["lora_model_path"] == "vertex_ai:job-9"
        assert "job-9" not in feature._finalized_jobs
        assert await w.active_handles() == ["comp-1:job-9"]
        assert provider.cleanup_calls == ["job-9"]

        # Second poll (reconciler retry): cleanup SUCCEEDS -> completed + guarded
        # + dropped from active_handles.
        second = await w.poll("comp-1:job-9")
        assert second.outcome is Outcome.DONE
        assert db.conn.rows["comp-1"]["lora_training_status"] == "completed"
        assert "job-9" in feature._finalized_jobs
        assert await w.active_handles() == []
        assert provider.cleanup_calls == ["job-9", "job-9"]

    @pytest.mark.asyncio
    async def test_cleanup_failure_bounded_by_max_attempts(self):
        """A permanently-erroring cleanup is capped: after MAX_CLEANUP_ATTEMPTS
        failures the job is forced to its terminal status + guard (loud log),
        instead of looping in 'finalizing' forever."""
        from kestrel_feature_visual.feature import MAX_CLEANUP_ATTEMPTS

        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        provider.provider_name = "vertex_ai"

        async def always_fail(job_id):
            provider.cleanup_calls.append(job_id)
            raise RuntimeError("permanent teardown error")  # not an "already gone" msg
        provider.cleanup = always_fail
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        # Poll repeatedly. Up to the cap it stays finalizing/unguarded.
        for i in range(MAX_CLEANUP_ATTEMPTS - 1):
            s = await w.poll("comp-1:job-9")
            assert s.outcome is Outcome.PENDING
            assert db.conn.rows["comp-1"]["lora_training_status"] == "finalizing"
            assert "job-9" not in feature._finalized_jobs

        # The MAX-th attempt forces terminal + guard despite cleanup still erroring.
        final = await w.poll("comp-1:job-9")
        assert final.outcome is Outcome.DONE
        assert db.conn.rows["comp-1"]["lora_training_status"] == "completed"
        assert "job-9" in feature._finalized_jobs
        assert await w.active_handles() == []
        assert len(provider.cleanup_calls) == MAX_CLEANUP_ATTEMPTS

    @pytest.mark.asyncio
    async def test_cleanup_already_gone_treated_as_success(self):
        """An 'already gone' cleanup error means the pod is already torn down ->
        treat as success so we don't loop forever on an already-clean pod."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        provider.provider_name = "vertex_ai"
        provider.cleanup_raises = True  # raises "session already torn down"
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.DONE
        assert db.conn.rows["comp-1"]["lora_training_status"] == "completed"
        assert "job-9" in feature._finalized_jobs

    @pytest.mark.asyncio
    async def test_failed_job_cleanup_failure_stays_finalizing(self):
        """FAILED terminal follows the same shape: finalizing -> (on cleanup
        success) failed + guard; on cleanup failure stays finalizing."""
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_provider": "vertex_ai"},
        })
        provider = FakeTrainingProvider([FakeStatus(TS.FAILED, error="boom")])
        provider.provider_name = "vertex_ai"
        state = {"fail": True}

        async def flaky_cleanup(job_id):
            provider.cleanup_calls.append(job_id)
            if state["fail"]:
                state["fail"] = False
                raise RuntimeError("connection reset")
        provider.cleanup = flaky_cleanup
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        first = await w.poll("comp-1:job-9")
        assert first.outcome is Outcome.PENDING
        assert db.conn.rows["comp-1"]["lora_training_status"] == "finalizing"
        assert "job-9" not in feature._finalized_jobs

        second = await w.poll("comp-1:job-9")
        assert second.outcome is Outcome.FAILED
        assert db.conn.rows["comp-1"]["lora_training_status"] == "failed"
        assert "job-9" in feature._finalized_jobs


# ---------------------------------------------------------------------------
# Restart status-lookup crash (codex P2): get_status can RAISE for a persisted
# job after a restart (session-based provider recreated without _active_jobs).
# poll() must degrade cleanly — PENDING (retry) for the first attempts, FAILED
# after the bound, and a successful read resets the per-job counter.
# ---------------------------------------------------------------------------

class TestStatusUnknownAfterRestart:
    @pytest.mark.asyncio
    async def test_get_status_raise_is_pending_then_failed(self):
        from kestrel_feature_visual.wait_provider import MAX_STATUS_UNKNOWN_ATTEMPTS

        # No persisted lora_model_path: the job genuinely never completed, so
        # after the cap it must finalize FAILED.
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        provider = FakeTrainingProvider([])  # statuses unused; get_status overridden
        provider.provider_name = "vertex_ai"

        async def boom(job_id):
            raise RuntimeError("no such session after restart")
        provider.get_status = boom
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        # First N-1 polls: transient unknown -> PENDING (not propagated).
        for i in range(MAX_STATUS_UNKNOWN_ATTEMPTS - 1):
            s = await w.poll("comp-1:job-9")
            assert s.outcome is Outcome.PENDING
            assert "status unavailable for job-9" in s.summary
            assert s.data["status_error"] == "no such session after restart"
            assert s.data["status_unknown_attempts"] == i + 1
            # No terminal write yet; job stays enumerated.
            assert db.conn.rows["comp-1"]["lora_training_status"] == "running"
            assert "job-9" not in feature._finalized_jobs

        # The bound-th attempt: terminal FAILED so the job stops being enumerated.
        final = await w.poll("comp-1:job-9")
        assert final.outcome is Outcome.FAILED
        assert "status unrecoverable for job-9 after restart" in final.summary
        assert final.data["status_error"] == "no such session after restart"
        # Counter cleared after the terminal decision.
        assert feature._status_unknown_attempts.get("job-9") is None
        # codex round 6: the terminal FAILED must be PERSISTED + guarded, else
        # active_handles would keep enumerating this job forever.
        assert db.conn.rows["comp-1"]["lora_training_status"] == "failed"
        assert "job-9" in feature._finalized_jobs
        # And it is no longer enumerated as in-flight.
        assert await w.active_handles() == []

    @pytest.mark.asyncio
    async def test_completed_job_recovered_not_marked_failed(self):
        """codex P2 data-correctness: a job that already COMPLETED (its
        ``lora_model_path`` is persisted and status is "finalizing" because the
        final finalizing->completed write failed transiently) must NOT be
        overwritten FAILED when get_status later raises (session cleaned up).
        After the cap, poll() recovers it as COMPLETED from the persisted
        metadata."""
        from kestrel_feature_visual.wait_provider import MAX_STATUS_UNKNOWN_ATTEMPTS

        model_path = "gs://bucket/loras/comp-1/weights.safetensors"
        db = FakeDbPool(rows={
            "comp-1": {
                "lora_training_status": "finalizing",
                "lora_job_id": "job-9",
                "lora_trigger_word": "TOK",
                "lora_provider": "vertex_ai",
                "lora_model_path": model_path,
            },
        })
        provider = FakeTrainingProvider([])  # statuses unused; get_status overridden
        provider.provider_name = "vertex_ai"

        async def boom(job_id):
            raise RuntimeError("no such session after restart")
        provider.get_status = boom
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        # First N-1 polls: transient unknown -> PENDING; job stays finalizing.
        for i in range(MAX_STATUS_UNKNOWN_ATTEMPTS - 1):
            s = await w.poll("comp-1:job-9")
            assert s.outcome is Outcome.PENDING
            assert db.conn.rows["comp-1"]["lora_training_status"] == "finalizing"
            assert "job-9" not in feature._finalized_jobs

        # The cap-th poll: recover the already-completed job as DONE, NOT FAILED.
        final = await w.poll("comp-1:job-9")
        assert final.outcome is Outcome.DONE
        assert "recovered from persisted metadata" in final.summary
        assert final.data["lora_path"] == model_path
        # Persisted status is now "completed" and the path is preserved.
        assert db.conn.rows["comp-1"]["lora_training_status"] == "completed"
        assert db.conn.rows["comp-1"]["lora_model_path"] == model_path
        # Job is finalized + no longer enumerated as in-flight.
        assert "job-9" in feature._finalized_jobs
        assert await w.active_handles() == []

    @pytest.mark.asyncio
    async def test_successful_get_status_resets_counter(self):
        TS = _training_state()
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOK", "lora_provider": "vertex_ai"},
        })
        provider = FakeTrainingProvider([])
        provider.provider_name = "vertex_ai"

        # Fail twice, then succeed (TRAINING) on the third call.
        calls = {"n": 0}

        async def flaky(job_id):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise RuntimeError("lost session")
            return FakeStatus(TS.TRAINING, progress=0.6)
        provider.get_status = flaky
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        s1 = await w.poll("comp-1:job-9")
        assert s1.outcome is Outcome.PENDING
        assert feature._status_unknown_attempts["job-9"] == 1
        s2 = await w.poll("comp-1:job-9")
        assert s2.outcome is Outcome.PENDING
        assert feature._status_unknown_attempts["job-9"] == 2

        # Third poll: get_status succeeds -> normal PENDING (training) AND the
        # transient-unknown counter is cleared.
        s3 = await w.poll("comp-1:job-9")
        assert s3.outcome is Outcome.PENDING
        assert s3.data["state"] == "training"
        assert "job-9" not in feature._status_unknown_attempts


class TestLazyTrainingRecordsInflight:
    """codex round 7: the lazy generate_selfie(allow_training=True) dispatch
    path (_get_or_train_lora) must ALSO record in-flight metadata, else the
    job is invisible to active_handles and the reconciler never finalizes it."""

    @pytest.mark.asyncio
    async def test_get_or_train_lora_records_inflight(self):
        import types as _types

        provider = FakeTrainingProvider([FakeStatus(_training_state().TRAINING)])
        provider.provider_name = "vertex_ai"

        async def fake_start_training(*, companion_id, avatar_data, config):
            return _types.SimpleNamespace(
                job_id="lazy-job-1",
                provider="vertex_ai",
                trigger_word=getattr(config, "trigger_word", None),
                output_path=None,
            )
        provider.start_training = fake_start_training

        db = FakeDbPool(rows={})
        # The lazy path's avatar lookup uses a different SELECT shape; give the
        # fake conn an avatar_data row for this companion.
        db.conn.rows["comp-7"] = {}
        async def fetchrow(sql, *params):
            if "avatar_data" in sql:
                return {"avatar_data": b"img-bytes", "image_url": None}
            cid = params[0]
            return {"avatar_config": db.conn.rows.get(cid)} if cid in db.conn.rows else None
        db.conn.fetchrow = fetchrow

        feature = make_feature(provider, db)
        feature._get_training_provider = lambda name=None: provider

        result = await feature._get_or_train_lora("comp-7")

        assert result == "training:lazy-job-1"
        # In-flight metadata persisted so active_handles can enumerate it.
        cfg = db.conn.rows["comp-7"]
        assert cfg["lora_training_status"] == "running"
        assert cfg["lora_job_id"] == "lazy-job-1"
        assert cfg["lora_provider"] == "vertex_ai"


class TestTriggerWordKeyParity:
    """codex round 10: the trigger must be persisted under BOTH lora_trigger_word
    (finalizer's own recovery key) AND trigger_word (the key the selfie
    generation path reads), else a custom trigger never activates the LoRA."""

    @pytest.mark.asyncio
    async def test_finalize_persists_trigger_under_both_keys(self):
        TS = _training_state()
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED)])
        db = FakeDbPool(rows={
            "comp-1": {"lora_training_status": "running", "lora_job_id": "job-9",
                       "lora_trigger_word": "TOKalice", "lora_provider": "fake_runpod"},
        })
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")
        assert status.outcome is Outcome.DONE
        cfg = db.conn.rows["comp-1"]
        assert cfg["lora_trigger_word"] == "TOKalice"
        # The generation-path key must also be set to the same value.
        assert cfg["trigger_word"] == "TOKalice"


class TestFinalizeRetryPreservesPath:
    """codex round 11: a finalize RETRY (cleanup failed first time) without
    provider output details must NOT overwrite the already-persisted
    lora_model_path with a provider:job_id sentinel."""

    @pytest.mark.asyncio
    async def test_retry_without_output_details_keeps_persisted_path(self):
        TS = _training_state()
        real_path = "gs://bucket/comp-1/pytorch_lora_weights.safetensors"
        # Row already in 'finalizing' with the real path persisted (first pass
        # completed but cleanup failed, leaving it finalizing).
        db = FakeDbPool(rows={
            "comp-1": {
                "lora_training_status": "finalizing",
                "lora_job_id": "job-9",
                "lora_model_path": real_path,
                "lora_trigger_word": "TOKalice",
                "lora_provider": "fake_runpod",
            },
        })
        # Status-only snapshot: COMPLETED but NO provider_details / output_path.
        provider = FakeTrainingProvider([FakeStatus(TS.COMPLETED, provider_details={})])
        feature = make_feature(provider, db)
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")
        assert status.outcome is Outcome.DONE
        # The persisted real path must survive — NOT be clobbered by a sentinel.
        assert db.conn.rows["comp-1"]["lora_model_path"] == real_path
        assert "fake_runpod:job-9" not in str(db.conn.rows["comp-1"]["lora_model_path"])
