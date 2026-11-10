"""
A simulação fica viva depois da última rodada (modo de espera de comandos,
para entrevistas). O status COMPLETED precisa ser publicado sem esperar o
processo encerrar, mas só depois de drenar a gravação no Zep.
"""

import pytest

from app.services import simulation_runner as runner_module
from app.services.simulation_runner import (
    RunnerStatus,
    SimulationRunState,
    SimulationRunner,
)


class FakeProcess:
    """Processo que continua vivo até `exit_after` chamadas de poll()."""

    pid = 4321

    def __init__(self, exit_after=None):
        self.exit_after = exit_after
        self.polls = 0
        self.returncode = None
        self.terminated = False

    def poll(self):
        self.polls += 1
        if self.terminated or (
            self.exit_after is not None and self.polls > self.exit_after
        ):
            self.returncode = 0
            return 0
        return None


def _write_platform_end(tmp_path, simulation_id):
    for platform in ("twitter", "reddit"):
        folder = tmp_path / simulation_id / platform
        folder.mkdir(parents=True)
        (folder / "actions.jsonl").write_text(
            '{"event_type":"simulation_end","total_rounds":10,"total_actions":3}\n',
            encoding="utf-8",
        )


@pytest.fixture
def runner_env(monkeypatch, tmp_path):
    saved = []
    synced = []
    state_holder = {}

    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        SimulationRunner,
        "get_run_state",
        classmethod(lambda _cls, _sid: state_holder.get("state")),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_save_run_state",
        classmethod(lambda _cls, s: saved.append(s.runner_status)),
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(lambda _cls, _sid, status, *_a, **_k: synced.append(status)),
    )
    monkeypatch.setattr(runner_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager,
        "get_updater",
        classmethod(lambda _cls, _sid: None),
    )

    yield {
        "tmp_path": tmp_path,
        "saved": saved,
        "synced": synced,
        "state_holder": state_holder,
    }

    for sid in list(SimulationRunner._processes):
        if sid.startswith("sim-wait"):
            SimulationRunner._processes.pop(sid, None)
            SimulationRunner._graph_memory_enabled.pop(sid, None)
            SimulationRunner._manual_stop_requests.discard(sid)
            SimulationRunner._monitor_threads.pop(sid, None)


def test_completed_is_published_while_process_waits_for_commands(
    runner_env, monkeypatch
):
    sid = "sim-wait-1"
    _write_platform_end(runner_env["tmp_path"], sid)
    state = SimulationRunState(simulation_id=sid, runner_status=RunnerStatus.RUNNING)
    runner_env["state_holder"]["state"] = state

    drained = []
    observed_while_alive = []
    process = FakeProcess(exit_after=3)

    def stop_updater(_cls, _sid):
        assert process.returncode is None, "dreno deve ocorrer com o processo vivo"
        drained.append(_sid)

    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager, "stop_updater", classmethod(stop_updater)
    )
    original_complete = SimulationRunner._complete_while_waiting_for_commands.__func__

    def spy(cls, simulation_id, st):
        result = original_complete(cls, simulation_id, st)
        observed_while_alive.append((result.runner_status, process.returncode))
        return result

    monkeypatch.setattr(
        SimulationRunner, "_complete_while_waiting_for_commands", classmethod(spy)
    )

    SimulationRunner._processes[sid] = process
    SimulationRunner._graph_memory_enabled[sid] = True

    SimulationRunner._monitor_simulation(sid)

    assert drained == [sid]
    assert observed_while_alive[0] == (RunnerStatus.COMPLETED, None)
    assert RunnerStatus.STOPPING in runner_env["synced"]
    assert state.runner_status == RunnerStatus.COMPLETED
    assert state.twitter_running is False and state.reddit_running is False
    # o dreno roda uma vez só, mesmo após o processo sair
    assert sid not in SimulationRunner._graph_memory_enabled


def test_drain_failure_publishes_failed(runner_env, monkeypatch):
    sid = "sim-wait-2"
    state = SimulationRunState(simulation_id=sid, runner_status=RunnerStatus.RUNNING)
    runner_env["state_holder"]["state"] = state
    monkeypatch.setattr(
        runner_module.ZepGraphMemoryManager,
        "stop_updater",
        classmethod(lambda _cls, _sid: (_ for _ in ()).throw(RuntimeError("zep off"))),
    )
    SimulationRunner._graph_memory_enabled[sid] = True

    result = SimulationRunner._complete_while_waiting_for_commands(sid, state)

    assert result.runner_status == RunnerStatus.FAILED
    assert "zep off" in result.error


def test_manual_stop_in_progress_is_not_overwritten(runner_env, monkeypatch):
    sid = "sim-wait-3"
    state = SimulationRunState(simulation_id=sid, runner_status=RunnerStatus.RUNNING)
    runner_env["state_holder"]["state"] = state
    SimulationRunner._manual_stop_requests.add(sid)

    result = SimulationRunner._complete_while_waiting_for_commands(sid, state)

    assert result.runner_status == RunnerStatus.RUNNING
    assert runner_env["saved"] == []


def test_completed_survives_normal_process_exit(runner_env, monkeypatch):
    sid = "sim-wait-4"
    _write_platform_end(runner_env["tmp_path"], sid)
    state = SimulationRunState(simulation_id=sid, runner_status=RunnerStatus.RUNNING)
    runner_env["state_holder"]["state"] = state
    process = FakeProcess(exit_after=2)
    SimulationRunner._processes[sid] = process

    SimulationRunner._monitor_simulation(sid)

    assert state.runner_status == RunnerStatus.COMPLETED
    assert state.error is None


def test_stop_accepts_completed_simulation_with_live_process(runner_env, monkeypatch):
    sid = "sim-wait-5"
    state = SimulationRunState(simulation_id=sid, runner_status=RunnerStatus.COMPLETED)
    runner_env["state_holder"]["state"] = state
    process = FakeProcess()
    SimulationRunner._processes[sid] = process
    monkeypatch.setattr(
        SimulationRunner,
        "_terminate_process",
        classmethod(lambda _cls, proc, _sid, **_k: setattr(proc, "terminated", True)),
    )

    result = SimulationRunner.stop_simulation(sid)

    assert process.terminated is True
    assert result.runner_status == RunnerStatus.STOPPED


def test_stop_still_rejects_completed_simulation_without_process(runner_env):
    sid = "sim-wait-6"
    state = SimulationRunState(simulation_id=sid, runner_status=RunnerStatus.COMPLETED)
    runner_env["state_holder"]["state"] = state

    with pytest.raises(ValueError):
        SimulationRunner.stop_simulation(sid)
