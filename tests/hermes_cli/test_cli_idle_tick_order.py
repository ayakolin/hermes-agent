"""The classic CLI's idle tick serves a waiting gateway turn before MCP housekeeping, and a dispatcher session
never auto-reloads MCP (its tool surface is frozen at creation, so a reload would break the prompt cache)."""
import queue
from unittest.mock import MagicMock

from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin


class _CLI(CLITuiRuntimeMixin):
    def __init__(self, head=None):
        self._pending_input = queue.Queue()
        if head is not None:
            self._pending_input.put(head)
        self.calls = []

    def _drain_process_notifications(self, consumer):
        self.calls.append("drain")

    def _check_termios_drift(self):
        self.calls.append("termios")

    def _maybe_fire_loop_tick(self):
        self.calls.append("loop")

    def _maybe_resume_parked_goal(self):
        self.calls.append("goal")

    def _check_config_mcp_changes(self):
        self.calls.append("mcp")


def test_the_idle_tick_drains_the_gateway_turn_before_mcp_housekeeping():
    cli = _CLI()
    cli._tui_idle_tick()
    assert cli.calls == ["drain", "termios", "loop", "goal", "mcp"]


def test_a_queued_turn_skips_the_slow_housekeeping_this_tick():
    from tools.process_registry_notifications import TimelineNotification
    cli = _CLI(head=TimelineNotification("text", "display", "process_complete"))
    cli._tui_idle_tick()
    assert cli.calls == ["drain"], "the reply the user waits for runs before an MCP reload"


def test_a_raising_check_never_escapes():
    class _Boom(_CLI):
        def _check_config_mcp_changes(self):
            super()._check_config_mcp_changes()
            raise RuntimeError("state.db locked")
    cli = _Boom()
    cli._tui_idle_tick()  # must not raise: process_loop logs 'msg may be lost' and moves on
    assert cli.calls == ["drain", "termios", "loop", "goal", "mcp"]


def test_a_dispatcher_session_only_gets_the_reload_notice(tmp_path):
    """A dispatcher's tool surface is frozen at creation: an auto reload would rebuild it mid-conversation."""
    from unittest.mock import patch

    import hermes_yaml as yaml

    from tests.hermes_cli.test_cli_mcp_config_watch import _make_cli

    for dispatcher in (False, True):
        obj, cfg_file = _make_cli(tmp_path, mcp_servers={})
        obj._busy_dispatch = dispatcher
        cfg_file.write_text(yaml.safe_dump({"mcp_servers": {"github": {"url": "https://mcp.github.com"}}}))
        obj._config_sig = None
        with patch("hermes_cli.config.get_config_path", return_value=cfg_file):
            obj._check_config_mcp_changes()
        assert obj._reload_mcp.called is (not dispatcher), (dispatcher, obj._reload_mcp.called)
