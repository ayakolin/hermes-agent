"""The classic CLI's busy input on a dispatcher session: one ordered path, a refusal that is queued, and the
owner's busy mode adopted from the server rather than from the CLI's own startup config.

Behaviour contracts only: no prompt_toolkit app, no sleeps (the worker's FIFO is waited on by draining it).
"""
import queue
from types import SimpleNamespace

import pytest

from hermes_cli.cli_busy_input_mixin import CLIBusyInputMixin
from hermes_cli.gateway_agent import STEERED, VOICE_STOPPED, Refusal, Unknown


class Shell(CLIBusyInputMixin):
    """A minimal classic CLI: the busy machinery plus what it reads."""

    def __init__(self, mode="interrupt"):
        self.busy_input_mode = mode
        self.agent = SimpleNamespace()
        self._agent_running = True
        self._pending_input = queue.Queue()
        self._interrupt_queue = queue.Queue()
        self._busy_dispatch = False
        self._busy_worker = self._busy_worker_thread = None
        self.printed = []
        self.enqueued = []
        self.store = object()  # a gateway-backed route: next-turn text rides the owner's durable FIFO

    def _busy_print(self, line):
        self.printed.append(line)

    def _busy_log_interrupt(self, payload):
        pass

    def _gateway_queue_store(self):
        return self.store

    def _queue_enqueue(self, text, *, running=None):
        self.enqueued.append((text, running))
        return True


def _dispatcher(mode="interrupt", **agent):
    shell = Shell(mode)
    shell._busy_dispatch = True
    shell.agent = SimpleNamespace(**{"_supports_active_turn_redirect": True, **agent})
    return shell


def _drain(shell):
    """Run the serial worker's pending items to completion, in order (no timing)."""
    if shell._busy_worker is not None:
        shell._busy_worker.join()


# ── the ladder ───────────────────────────────────────────────────────────────

def test_an_accepted_redirect_is_the_only_verb_tried():
    calls = []
    shell = _dispatcher(redirect=lambda text: calls.append(("redirect", text)) or True,
                        steer=lambda text: calls.append(("steer", text)) or True)
    shell._busy_submit("next", [], "next")
    _drain(shell)
    assert calls == [("redirect", "next")]


def test_a_declined_redirect_steers_before_it_queues():
    """redirect() declines whenever no model request is in flight (a burst's second message): a dispatcher
    steers next, so rapid-fire input is delivered instead of waiting behind the whole turn."""
    calls = []
    shell = _dispatcher(redirect=lambda text: calls.append("redirect") or False,
                        steer=lambda text: calls.append("steer") or True)
    shell._busy_submit("next", [], "next")
    _drain(shell)
    assert calls == ["redirect", "steer"]
    assert shell.enqueued == [] and shell._pending_input.empty()


def test_a_redirect_the_owner_took_as_a_steer_does_not_reanchor_the_ask():
    shell = _dispatcher(redirect=lambda text: STEERED)
    shell._busy_submit("next", [], "next")
    _drain(shell)
    assert shell.enqueued == [] and shell._interrupt_queue.empty()


def test_an_undelivered_refusal_queues_and_never_interrupts():
    shell = _dispatcher(redirect=lambda text: Refusal("agent_building"),
                        steer=lambda text: Refusal("steer_declined"))
    shell._busy_submit("next", [], "next")
    _drain(shell)
    assert shell.enqueued == [("next", True)]
    assert shell._interrupt_queue.empty(), "a dispatcher session must never write _interrupt_queue"


def test_an_unknown_outcome_is_neither_requeued_nor_interrupted():
    shell = _dispatcher(redirect=lambda text: Unknown("redirect_unconfirmed"))
    shell._busy_submit("next", [], "next")
    _drain(shell)
    assert shell.enqueued == [] and shell._interrupt_queue.empty()
    assert any("check /queue list" in line for line in shell.printed), shell.printed


def test_a_voice_stop_is_consumed():
    shell = _dispatcher(redirect=lambda text: VOICE_STOPPED)
    shell._busy_submit("stop", [], "stop")
    _drain(shell)
    assert shell.enqueued == [] and shell._interrupt_queue.empty()


def test_the_redirect_rung_falls_to_steer_when_the_agent_cannot_redirect():
    calls = []
    shell = _dispatcher(redirect=lambda text: calls.append("redirect") or True,
                        steer=lambda text: calls.append("steer") or True)
    shell.agent._supports_active_turn_redirect = False
    shell._busy_submit("next", [], "next")
    _drain(shell)
    assert calls == ["steer"] and shell.enqueued == []


def test_an_agent_without_redirect_queues_only_after_steer_declines():
    calls = []
    shell = _dispatcher(steer=lambda text: calls.append("steer") or False)
    shell.agent._supports_active_turn_redirect = False
    shell._busy_submit("next", [], "next")
    _drain(shell)
    assert calls == ["steer"] and shell.enqueued == [("next", True)]


# ── the classic modes are untouched ─────────────────────────────────────────

def test_the_classic_interrupt_mode_still_uses_the_interrupt_queue():
    shell = Shell("interrupt")
    shell.agent = SimpleNamespace(_supports_active_turn_redirect=True, redirect=lambda text: False)
    shell._busy_submit("next", [], "next")
    assert shell._interrupt_queue.get_nowait() == "next"


def test_the_classic_steer_mode_steers_first():
    shell = Shell("steer")
    shell.agent = SimpleNamespace(steer=lambda text: True)
    shell._busy_submit("next", [], "next")
    assert shell._pending_input.empty() and shell._interrupt_queue.empty()


def test_the_classic_queue_mode_uses_the_gateway_fifo():
    shell = Shell("queue")
    shell._busy_submit("next", [], "next")
    assert shell.enqueued == [("next", True)]


def test_a_steer_refusal_falls_back_to_the_queue():
    shell = Shell("steer")
    shell.agent = SimpleNamespace(steer=lambda text: Refusal("declined"))
    shell._busy_submit("next", [], "next")
    assert shell.enqueued == [("next", True)]


# ── ordering ────────────────────────────────────────────────────────────────

def test_the_worker_keeps_submission_order():
    seen = []
    shell = _dispatcher(redirect=lambda text: seen.append(text) or True)
    for text in ("one", "two", "three"):
        shell._busy_submit(text, [], text)
    _drain(shell)
    assert seen == ["one", "two", "three"]


def test_the_worker_uses_the_busy_state_captured_at_submit_time():
    """A message submitted while the turn ran queues where the user sent it — even if the worker only reaches
    it after the turn ended. Reading ``_agent_running`` then would send it to the local FIFO, behind anything
    the user typed since."""
    shell = _dispatcher(redirect=lambda text: Refusal("stale"))
    shell._busy_submit("typed first", [], "typed first")
    shell._agent_running = False
    _drain(shell)
    assert shell.enqueued == [("typed first", True)]


def test_a_burst_does_not_block_the_key_thread_on_the_rpc():
    """Submitting returns before the worker runs the correction (the ladder's RPC is 30 s + 5 s at worst)."""
    gate = _Gate()
    shell = _dispatcher(redirect=gate.wait_and_answer)
    shell._busy_submit("next", [], "next")
    assert shell._busy_worker.qsize() == 1 or shell._busy_worker.unfinished_tasks == 1
    gate.release()
    _drain(shell)


class _Gate:
    """A redirect that blocks until the test releases it, proving submit returned first."""

    def __init__(self):
        import threading
        self.entered, self._go, self._released = threading.Event(), threading.Event(), threading.Event()

    def wait_and_answer(self):
        self.entered.set()
        self._go.wait(10)
        self._released.set()
        return True

    def release(self):
        self._go.set()
        self._released.wait(10)


# ── the server-resolved mode ────────────────────────────────────────────────

def test_the_cli_adopts_the_server_resolved_busy_mode_and_route():
    """A dispatcher session ignores the CLI's own startup config here too: the owner's ``config.get busy``
    names both the mode and whether its corrections take the ordered ladder."""
    shell = Shell("queue")  # display.busy_input_mode in the CLI's own config
    shell._adopt_server_busy({"key": "busy", "value": "interrupt", "dispatch": True})
    assert shell.busy_input_mode == "interrupt" and shell._busy_dispatches()
    shell._adopt_server_busy({"key": "busy", "value": "steer"})  # a classic session: no dispatch flag
    assert shell.busy_input_mode == "steer" and not shell._busy_dispatches()


def test_a_bad_value_never_overwrites_the_mode():
    shell = Shell("steer")
    shell._adopt_server_busy({"key": "busy", "value": "bogus"})
    assert shell.busy_input_mode == "steer"


def test_without_a_gateway_store_the_cli_keeps_its_local_mode():
    shell = Shell("steer")
    shell.store = None
    shell._mirror_server_busy()
    assert shell.busy_input_mode == "steer" and not shell._busy_dispatches()
