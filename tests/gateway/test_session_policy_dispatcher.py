"""Dispatcher mode in a local session's frozen launch policy (gateway/session_policy*.py).

``stage_local_session`` is the one caller that opts a new local session into dispatcher mode, and
``build_policy`` freezes the answer in ``LocalSessionPolicy.dispatcher`` together with the toolsets it
produced. A local session carries its decision in the policy, not in a session-row marker: every turn and
every rebuild reads it there, so a later config edit never changes a running conversation's tool array.
"""

from __future__ import annotations

import pytest

from agent.dispatcher_session import DISPATCHER_TOOLSET


def create(params, config, **kwargs):
    """The ``session.create`` build (``stage_local_session``), the one caller that opts in."""
    from gateway.session_policy import build_policy
    return build_policy(params, config, workspace_posture=True, dispatcher_session=True, **kwargs)


@pytest.fixture
def cwd(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    return folder


@pytest.mark.parametrize("source", ["cli", "tui", "gui"])
def test_an_interactive_local_session_runs_the_dispatcher_posture(source, cwd):
    policy = create({"source": source, "cwd": str(cwd)}, {})
    assert policy.dispatcher is True
    assert DISPATCHER_TOOLSET in policy.toolsets
    # The client surface's own toolsets still fold in: they are the client's, not a model capability.
    folded = {"cli": set(), "tui": {"project"}, "gui": {"desktop_ui", "catalog", "project"}}[source]
    assert folded <= set(policy.toolsets)


def test_the_profile_default_is_replaced_and_operator_extras_are_appended(cwd):
    """A dispatcher does not run the profile's default toolset selection; ``dispatcher.extra_toolsets`` is
    the one place an operator widens it, and only at creation."""
    policy = create({"source": "cli", "cwd": str(cwd)}, {"dispatcher": {"extra_toolsets": ["web"]}})
    assert policy.dispatcher is True
    assert set(policy.toolsets) == {DISPATCHER_TOOLSET, "web"}


def test_the_toolsets_stay_frozen_when_the_profile_changes(cwd):
    """``current_toolsets`` reads the profile's selection every turn for a following session; a dispatcher
    must not, or an edit mid-conversation would flip its tools[] and break the prompt cache."""
    from dataclasses import replace

    policy = create({"source": "tui", "cwd": str(cwd), "follow_profile_config": True}, {})
    frozen = policy.toolsets
    assert policy.current_toolsets({"platform_toolsets": {"cli": ["terminal"]}}) == frozen
    # The control: the same session without dispatcher mode follows the profile again.
    control = replace(policy, dispatcher=False)
    assert "terminal" in control.current_toolsets({"platform_toolsets": {"cli": ["terminal"]}})


@pytest.mark.parametrize("params, config", [
    ({"source": "oneshot"}, {}),
    ({"source": "acp"}, {}),
    ({"source": "tool"}, {}),
    ({"source": "cli", "toolsets": ["terminal"]}, {}),
    # A client that sent its own shell environment was proven local, and such a client cannot receive a
    # background result: its session keeps the surface it launched with.
    ({"source": "cli", "client_env": {"PATH": "/client/bin"}}, {}),
    ({"source": "cli", "safe_mode": True, "model": "m"}, {}),
    ({"source": "cli"}, {"dispatcher": {"enabled": False}}),
    ({"source": "cli"}, {"agent": {"coding_context": "focus"}}),
    ({"source": "cli"}, {"agent": {"disabled_toolsets": ["delegation"]}}),
    # A saved ``platform_toolsets.cli`` is what ``hermes tools`` / ``profiles.configure`` wrote: the
    # operator's selection wins over the default-on dispatcher.
    ({"source": "cli"}, {"platform_toolsets": {"cli": ["terminal"]}}),
    # A managed worker executes out of process and exits at its turn boundary: no later result turn.
    ({"source": "cli"}, {"gateway": {"managed_workers": True}}),
])
def test_every_opt_out_keeps_the_classic_policy(params, config, cwd):
    policy = create({**params, "cwd": str(cwd)}, config)
    assert policy.dispatcher is False
    assert DISPATCHER_TOOLSET not in policy.toolsets


def test_only_the_opting_in_caller_gets_dispatcher_mode(cwd):
    """Resume migration, cron, hosted rooms and bypass launches build a policy without the opt-in."""
    from gateway.session_policy import build_policy
    assert build_policy({"source": "cli", "cwd": str(cwd)}, {}).dispatcher is False


def test_a_side_agent_of_a_dispatcher_session_keeps_the_classic_toolsets(cwd):
    """``/bg``, ``/refine`` and the preview restart do the work a dispatcher hands off, so they need the
    terminal and the skill tools its own surface lacks."""
    policy = create({"source": "tui", "cwd": str(cwd)}, {})
    classic = policy.classic_toolsets(policy.config())
    assert "terminal" in classic and "skills" in classic
    assert DISPATCHER_TOOLSET not in classic
    assert set(policy.current_toolsets(policy.config())) == set(policy.toolsets)


def test_a_classic_policy_has_one_selection(cwd):
    from gateway.session_policy import build_policy
    policy = build_policy({"source": "tui", "cwd": str(cwd), "toolsets": ["terminal"]}, {})
    assert policy.dispatcher is False
    assert policy.classic_toolsets(policy.config()) == policy.current_toolsets(policy.config())


def test_a_stored_policy_round_trips_and_validates_the_field(cwd):
    from dataclasses import asdict

    from gateway.session_policy import restore_policy
    from hermes_state_runtime import RuntimeStoreError

    policy = create({"source": "cli", "cwd": str(cwd), "model": "m"}, {})
    restored = restore_policy(asdict(policy))
    assert restored.dispatcher is True
    with pytest.raises(RuntimeStoreError):
        restore_policy({**asdict(policy), "dispatcher": "yes"})
