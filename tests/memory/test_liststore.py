"""EC-G-22 (dapr-agents#885): per-id isolation in ConversationListMemory."""
import pytest

from dapr_agents.memory.liststore import ConversationListMemory


def _msg(text):
    return {"role": "user", "content": text}


def test_two_ids_do_not_see_each_other():
    mem = ConversationListMemory()
    mem.add_message(_msg("hello w1"), "w1")
    mem.add_messages([_msg("a"), _msg("b")], "w2")
    got1 = mem.get_messages("w1")
    got2 = mem.get_messages("w2")
    assert [m["content"] for m in got1] == ["hello w1"]
    assert [m["content"] for m in got2] == ["a", "b"]


def test_broadcast_only_returns_broadcast():
    mem = ConversationListMemory()
    mem.add_message(_msg("run summary"), "w1")
    mem.add_message(_msg("shared note"), "broadcast")
    got = mem.get_messages("broadcast")
    assert [m["content"] for m in got] == ["shared note"]


def test_reset_only_clears_given_id():
    mem = ConversationListMemory()
    mem.add_message(_msg("x"), "w1")
    mem.add_message(_msg("y"), "w2")
    mem.reset_memory("w1")
    assert mem.get_messages("w1") == []
    assert [m["content"] for m in mem.get_messages("w2")] == ["y"]


def test_unknown_id_returns_empty():
    assert ConversationListMemory().get_messages("nope") == []
