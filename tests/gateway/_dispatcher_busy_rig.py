"""Shared rig for the dispatcher busy-path tests: a real GatewayRunner, WhatsAppAdapter and SessionAuthority
over SQLite in a temp HERMES_HOME whose config.yaml carries the policy under test. The running turn is a
stand-in agent on the session's turn slot, so what is exercised is the real resolution, ladder and admission."""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from hermes_state_runtime import list_session_admissions

USER = '15551234567@s.whatsapp.net'


class StandInAgent:
    """What the busy path sees of a running AIAgent."""

    def __init__(self, *, dispatcher=True, redirect=True, children=()):
        self._dispatcher_session = dispatcher
        self._supports_active_turn_redirect = redirect
        self._active_children = list(children)
        self.redirects, self.steers, self.interrupts = [], [], []
        self.redirect_result = True
        self.steer_pending = False
        self.session_id = 'stand-in'

    def redirect(self, text):
        self.redirects.append(text)
        return self.redirect_result

    def steer(self, text):
        self.steers.append(text)
        return True

    def interrupt(self, *args):
        self.interrupts.append(args)

    def has_pending_steer(self):
        return self.steer_pending

    def _has_pending_redirect(self):
        return False

    def get_activity_summary(self):
        return {}


@dataclass
class Rig:
    runner: GatewayRunner
    authority: SessionAuthority
    adapter: object
    source: object
    key: str
    agent: StandInAgent
    sent: list = field(default_factory=list)
    _tasks: list = field(default_factory=list)

    def event(self, text='hello', message_id='m1', reply_expected=True, **kw):
        return MessageEvent(text=text, source=self.source, message_id=message_id,
                            reply_expected=reply_expected, **kw)

    def admitted(self):
        return [row for sid in self.authority.sessions
                for row in list_session_admissions(self.authority.db, session_id=sid, pending_only=False)]

    @property
    def turn(self):
        return self.runner._session_state(self.key).turn

    async def handle(self, event):
        """Run the runner's handler; True when it returned (the text landed in the running turn), False when
        it is parked behind the running turn (queued)."""
        before = len(self.admitted())
        task = asyncio.ensure_future(self.runner._handle_message(event))
        self._tasks.append(task)
        return await _first_of(task, lambda: len(self.admitted()) > before)

    async def close(self):
        pending = [*self._tasks, *self.adapter._background_tasks, *self.runner._background_tasks]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self.authority.db.close()


async def _first_of(task, queued):
    """True when ``task`` finished first, False when a row was admitted (parked) while it still runs."""
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: task.done() or queued(), timeout=10, what='the message to land or park')
    return task.done()


def reacting_adapter_class(acknowledges=True):
    """A WhatsApp adapter whose processing hooks record into ``calls`` and that says whether it reacts."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    class Reacting(WhatsAppAdapter):
        calls = None

        def acknowledges_event(self, event):
            return acknowledges

        async def on_processing_start(self, event):
            self.calls.append(('start', event.message_id))

        async def on_processing_complete(self, event, outcome):
            self.calls.append(('complete', event.message_id, outcome))

    return Reacting


@asynccontextmanager
async def rig(tmp_path, monkeypatch, config=None, agent=None, adapter_cls=None, drain=False):
    """``drain=True`` leaves the authority's drain running and the turn slot empty, so a test starts a real
    turn through the adapter and a queued message runs when it ends."""
    from hermes_constants import get_hermes_home
    home = get_hermes_home()  # the conftest's isolated home: state.db is pinned there, so stay in it
    (home / 'config.yaml').write_text(json.dumps(config or {}), encoding='utf-8')  # JSON is valid YAML
    monkeypatch.setattr('gateway.run._hermes_home', home)
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    if not drain:
        monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # parked rows are never claimed
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = (adapter_cls or WhatsAppAdapter)(
        PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.calls = []
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='busy')
    runner._wire_adapter_handlers(adapter)
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append(SimpleNamespace(chat_id=chat_id, content=content, metadata=kwargs.get('metadata')))
        return SendResult(success=True, message_id='s')

    monkeypatch.setattr(adapter, 'send', send)
    source = adapter.build_source(chat_id=USER, user_id=USER, chat_type='group')
    key = runner._session_key_for_source(source)
    agent = agent or StandInAgent()
    built = Rig(runner, authority, adapter, source, key, agent, sent)
    if not drain:
        built.turn.agent = agent
        built.turn.event = MessageEvent(text='earlier question', source=source, message_id='t0', reply_expected=True)
        built.turn.started_ts = time.time()
    try:
        yield built
    finally:
        await built.close()
