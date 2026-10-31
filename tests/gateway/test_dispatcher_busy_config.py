"""``dispatcher.busy.*`` and ``display.dispatcher_receipt`` as the gateway reads them: raw YAML, no
DEFAULT_CONFIG merge, bad values named in a warning and replaced by the default."""
import logging

from gateway.run_busy_dispatcher import dispatcher_busy_from_config


def test_defaults_when_nothing_is_configured():
    cfg = dispatcher_busy_from_config({})
    assert cfg.settings.busy_policy == 'dispatch' and cfg.settings.quick_lane is True
    assert cfg.settings.stall_seconds == 4.0 and cfg.receipt == 'auto' and cfg.receipt_for('telegram') == 'auto'


def test_values_and_per_platform_receipt_overrides():
    cfg = dispatcher_busy_from_config({
        'dispatcher': {'busy': {'policy': 'inherit', 'quick_lane': False, 'stall_seconds': 9}},
        'display': {'dispatcher_receipt': 'text', 'platforms': {'telegram': {'dispatcher_receipt': 'reaction'}}},
    })
    assert (cfg.settings.busy_policy, cfg.settings.quick_lane, cfg.settings.stall_seconds) == ('inherit', False, 9.0)
    assert cfg.receipt_for('telegram') == 'reaction' and cfg.receipt_for('slack') == 'text'


def test_a_bad_value_is_named_and_replaced(caplog):
    with caplog.at_level(logging.WARNING, logger='gateway.run'):
        cfg = dispatcher_busy_from_config({
            'dispatcher': {'busy': {'policy': 'sometimes'}},
            'display': {'dispatcher_receipt': 'loud', 'platforms': {'slack': {'dispatcher_receipt': 'maybe'}}},
        })
    assert cfg.settings.busy_policy == 'dispatch' and cfg.receipt == 'auto' and cfg.receipt_for('slack') == 'auto'
    text = caplog.text
    assert 'dispatcher.busy.policy' in text and 'display.dispatcher_receipt' in text
    assert 'display.platforms.slack.dispatcher_receipt' in text


def test_unquoted_yaml_off_reads_as_off_through_the_real_loader(tmp_path):
    """YAML 1.1 reads an unquoted ``off`` as False; a user who writes it means off."""
    from gateway.run import _load_gateway_config
    config = tmp_path / 'config.yaml'
    config.write_text('display:\n  dispatcher_receipt: off\n  platforms:\n    telegram:\n      dispatcher_receipt: on\n',
                      encoding='utf-8')
    cfg = dispatcher_busy_from_config(_load_gateway_config(config))
    assert cfg.receipt == 'off' and cfg.receipt_for('telegram') == 'auto'
