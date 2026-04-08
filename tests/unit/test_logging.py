import logging

from src.utils.logging import get_logger


def test_returns_logger_instance():
    logger = get_logger("test.basic")
    assert isinstance(logger, logging.Logger)


def test_logger_name_matches():
    logger = get_logger("test.naming")
    assert logger.name == "test.naming"


def test_no_duplicate_handlers_on_repeated_calls():
    logger = get_logger("test.dedup")
    count_after_first_call = len(logger.handlers)

    get_logger("test.dedup")
    get_logger("test.dedup")

    assert len(logger.handlers) == count_after_first_call


def test_propagation_disabled():
    logger = get_logger("test.propagate")
    assert logger.propagate is False


def test_default_level_is_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    # Use a unique name so the handler-guard doesn't interfere
    logger = get_logger("test.defaultlevel")
    assert logger.level == logging.INFO


def test_logger_has_at_least_one_handler():
    logger = get_logger("test.hashandler")
    assert len(logger.handlers) >= 1
