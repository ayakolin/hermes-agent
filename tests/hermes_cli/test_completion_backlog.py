"""Completion backlogs preserve results without multiplying autonomous turns."""
from evals.completion_backlog_probe import probe


def test_ready_completions_share_one_turn_across_interactive_routes(tmp_path):
    for surface in ("cli", "poller", "post-turn"):
        for scenario in ("backlog", "single", "mixed"):
            result = probe(surface, scenario, tmp_path / surface / scenario)
            # A mixed run is completions + one delegation: the gateway and CLI keep the delegation in its own
            # turn (5), while the legacy poller and the post-turn drain group a delegation with the completions
            # that landed together into ONE turn (4). Every payload still arrives once, in order (asserted below).
            expected_turns = 1 if scenario != "mixed" else (5 if surface == "cli" else 4)
            assert result["wire_turns"] == expected_turns, result
            assert result["payload_order"] == sorted(result["payload_order"]), result
            if scenario == "mixed":
                assert result["delegation_delivered_once"], result
            assert result["all_payloads_preserved"], result
            if scenario == "single":
                assert result["single_exact"], result


def test_consumed_or_foreign_completions_never_start_a_turn(tmp_path):
    for surface in ("cli", "poller", "post-turn"):
        for scenario in ("consumed", "foreign"):
            result = probe(surface, scenario, tmp_path / surface / scenario)
            assert result["wire_turns"] == 0, result
            if scenario == "foreign":
                assert result["queue_remaining"] == result["children"], result
