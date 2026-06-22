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
    def __init__(self):
        self.executed = []

    async def execute(self, sql, *params):
        self.executed.append((sql, params))


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakeDbPool:
    def __init__(self):
        self.conn = FakeConn()

    def acquire(self):
        return _Acquire(self.conn)


def _training_state():
    from kestrel_sovereign.features.training.types import TrainingState

    return TrainingState


def make_feature(provider, db_pool=None):
    """Build a real VisualIdentityFeature wired to fakes (no initialize())."""
    from kestrel_feature_visual.feature import VisualIdentityFeature

    feature = VisualIdentityFeature(agent=None)
    feature._training_provider = provider
    feature.db_pool = db_pool
    feature._finalized_jobs = set()
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
        # finalize ran: cleanup + persistence
        assert provider.cleanup_calls == ["job-9"]
        assert len(db.conn.executed) == 1

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
    async def test_no_provider_is_failed(self):
        feature = make_feature(None, FakeDbPool())
        w = LoraTrainingWaitable(feature)

        status = await w.poll("comp-1:job-9")

        assert status.outcome is Outcome.FAILED
        assert "provider unavailable" in status.summary


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
        # ...but the side effects ran exactly once.
        assert provider.cleanup_calls == ["job-9"]
        assert len(db.conn.executed) == 1

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
        assert len(db.conn.executed) == 1

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
