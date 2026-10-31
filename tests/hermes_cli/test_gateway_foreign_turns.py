"""A notification turn the classic CLI missed the start of is painted from its receipt.

``message.start`` for a gateway-admitted automation turn is dropped when a local turn's drain or
``_drain_stale`` owns the stream: without recovery the user sees the turn AFTER it and never the
"background work finished" line that drove it. Real imports against a temp HERMES_HOME; no prompt.submit
is ever sent for the recovered turn (the owner already ran it).
"""
import asyncio
import queue
from types import SimpleNamespace

from hermes_cli import gateway_tools_notify
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with, frame
from tools.process_registry_notifications import TimelineNotification

TEXT = "[IMPORTANT: Background process proc_1 completed (exit 0).\nOutput:\nDONE]"
DISPLAY = "Background Process Finished: sleep 6; echo DONE"


class ReceiptClient(FakeClient):
    """``FakeClient`` with ``prompt.receipt`` answering from a per-admission map."""

    def __init__(self):
        super().__init__([])
        self.receipts = {}

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "prompt.receipt":
            return dict(self.receipts.get(params["admission_id"], {"status": "unknown"}))
        return await super().rpc(method, _timeout=_timeout, **params)


def _cli(receipts):
    client = ReceiptClient()
    client.receipts = receipts
    conn = GatewayConnection(connect=connect_with(client))
    conn.start(5)
    cli = SimpleNamespace(session_id=SID, _pending_input=queue.Queue(),
                          _session_db=GatewaySessionStore(connection=conn))
    return conn, cli, client


def _drop_start(cli, conn, admission="adm-9"):
    """The start arrives outside a turn and is dropped: remembered with its card."""
    from hermes_cli.gateway_tools_notify import missed_start
    missed_start(conn.state(SID), frame("message.start", {"display_kind": "process_complete",
                                                          "display_text": DISPLAY, "text": TEXT},
                                       generation=9, admission=admission)["params"])


def test_a_dropped_automation_start_is_painted_on_an_idle_tick():
    from hermes_cli.gateway_foreign_turns import recover_automation
    receipts = {}
    receipts["adm-9"] = {"status": "terminal", "outcome": "completed", "execution_generation": 9,
                                  "result": {"final_response": "done"}}
    conn, cli, client = _cli(receipts)
    try:
        _drop_start(cli, conn)
        recover_automation(cli, cli._session_db)
        painted = cli._pending_input.get_nowait()
        assert isinstance(painted, TimelineNotification) and painted.display_text == DISPLAY
        state = conn.state(SID)
        assert state.admitted_turn["admission_id"] == "adm-9" and state.admitted_turn["recovered"] is True
        assert state.admitted_turn["receipt"]["status"] == "terminal"
        assert state.missed_cards == {}, "a painted card is not recovered twice"
        assert not [m for m, _ in client.calls if m == "prompt.submit"]
    finally:
        conn.close()


def test_a_still_running_admission_is_adopted_without_a_receipt():
    from hermes_cli.gateway_foreign_turns import recover_automation
    receipts = {}
    receipts["adm-9"] = {"status": "started", "execution_generation": 9}
    conn, cli, client = _cli(receipts)
    try:
        _drop_start(cli, conn)
        recover_automation(cli, cli._session_db)
        adopted = conn.state(SID).admitted_turn
        assert adopted["recovered"] is True and "receipt" not in adopted
        assert adopted["execution_generation"] == 9
    finally:
        conn.close()


def test_an_admission_that_never_ran_is_forgotten():
    from hermes_cli.gateway_foreign_turns import recover_automation
    receipts = {}
    receipts["adm-9"] = {"status": "terminal", "outcome": "cancelled"}
    conn, cli, _ = _cli(receipts)
    try:
        _drop_start(cli, conn)
        recover_automation(cli, cli._session_db)
        assert cli._pending_input.empty() and conn.state(SID).missed_cards == {}
    finally:
        conn.close()


def test_one_recovered_turn_at_a_time():
    """The recovered header is still waiting for the input loop: the next card must not take its slot."""
    from hermes_cli.gateway_foreign_turns import recover_automation
    receipts = {}
    receipts["adm-9"] = {"status": "started", "execution_generation": 9}
    receipts["adm-10"] = {"status": "started", "execution_generation": 10}
    conn, cli, _ = _cli(receipts)
    try:
        _drop_start(cli, conn)
        _drop_start(cli, conn, admission="adm-10")
        recover_automation(cli, cli._session_db)
        recover_automation(cli, cli._session_db)  # the first header is still queued
        assert cli._pending_input.qsize() == 1
        assert set(conn.state(SID).missed_cards) == {"adm-10"}
        cli._pending_input.get_nowait()  # the input loop took it and ran the turn
        conn.state(SID).admitted_turn = None
        recover_automation(cli, cli._session_db)
        assert cli._pending_input.qsize() == 1
        assert conn.state(SID).missed_cards == {}
    finally:
        conn.close()


def test_the_adopted_turn_is_read_by_the_agent_and_survives_a_tick():
    """``take_admitted_turn`` hands the recovered turn to ``run_conversation``; the idle tick's stale-header
    reset leaves it alone while its header is still queued."""
    from hermes_cli.gateway_agent import GatewayAgent
    from hermes_cli.gateway_foreign_turns import recover_automation
    receipts = {}
    receipts["adm-9"] = {"status": "started", "execution_generation": 9}
    conn, cli, _ = _cli(receipts)
    try:
        _drop_start(cli, conn)
        recover_automation(cli, cli._session_db)
        agent = GatewayAgent(connection=conn, session_id=SID)
        gateway_tools_notify.queue_admitted_turn(cli, cli._session_db)  # the next idle tick
        assert conn.state(SID).admitted_turn is not None, "a queued recovered turn must survive the tick"
        adopted = gateway_tools_notify.take_admitted_turn(agent, TEXT)
        assert adopted["admission_id"] == "adm-9" and adopted["recovered"] is True
    finally:
        conn.close()


def test_a_start_the_stream_still_holds_is_not_remembered():
    """The normal path (``queue_admitted_turn`` reads it now) never fills the recovery map."""
    receipts = {}
    conn, cli, client = _cli(receipts)
    try:
        _drop_start(cli, conn)
        asyncio.run_coroutine_threadsafe(
            client.events.put(frame("message.start", {"display_kind": "process_complete", "display_text": DISPLAY,
                                                      "text": TEXT}, generation=9, admission="adm-9")),
            conn.loop).result(5)
        from tests.waiting import wait_until
        wait_until(lambda: conn.events.qsize() >= 1, timeout=5)
        gateway_tools_notify.queue_admitted_turn(cli, cli._session_db)
        assert conn.state(SID).missed_cards.get("adm-9") is None
        assert isinstance(cli._pending_input.get_nowait(), TimelineNotification)
    finally:
        conn.close()
