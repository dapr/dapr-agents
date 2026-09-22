#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from dapr_agents.storage.daprstores.stateservice import (
    StateStoreError,
    StateStoreService,
    _deep_merge,
)


class Widget(BaseModel):
    name: str
    count: int = 0


def _make_service(**kwargs) -> tuple[StateStoreService, MagicMock]:
    store = MagicMock()
    service = StateStoreService(
        store_name="widgets",
        store_factory=lambda: store,
        retry_initial_backoff=0,
        **kwargs,
    )
    return service, store


def _state_response(payload=None, etag=None):
    response = MagicMock()
    response.data = json.dumps(payload).encode() if payload is not None else b""
    response.etag = etag
    response.json = MagicMock(return_value=payload)
    return response


def _bulk_item(key, payload):
    item = MagicMock()
    item.key = key
    item.data = json.dumps(payload) if payload is not None else ""
    return item


def test_constructor_requires_store_name():
    with pytest.raises(StateStoreError):
        StateStoreService(store_name="")


def test_constructor_clamps_retry_params():
    service, _ = _make_service(
        retry_attempts=0, retry_backoff_multiplier=0.1, retry_jitter=-1
    )
    assert service.retry_attempts == 1
    assert service.retry_backoff_multiplier == 1.0
    assert service.retry_jitter == 0.0


def test_qualify_and_strip_prefix_round_trip():
    service, _ = _make_service(key_prefix="blog:")
    assert service._qualify("post-1") == "blog:post-1"
    assert service._strip_prefix("blog:post-1") == "post-1"


def test_strip_prefix_is_noop_when_prefix_missing():
    service, _ = _make_service(key_prefix="blog:")
    assert service._strip_prefix("post-1") == "post-1"


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"name": "a"}, {"name": "a"}),
        ('{"name": "a"}', {"name": "a"}),
        (b'{"name": "a"}', {"name": "a"}),
    ],
)
def test_ensure_dict_accepts_dict_str_and_bytes(value, expected):
    service, _ = _make_service()
    assert service._ensure_dict(value) == expected


def test_ensure_dict_accepts_pydantic_model():
    service, _ = _make_service()
    assert service._ensure_dict(Widget(name="a", count=2)) == {
        "name": "a",
        "count": 2,
    }


def test_ensure_dict_rejects_invalid_json_string():
    service, _ = _make_service()
    with pytest.raises(StateStoreError):
        service._ensure_dict("not json")


def test_ensure_dict_rejects_non_dict_json():
    service, _ = _make_service()
    with pytest.raises(StateStoreError):
        service._ensure_dict("[1, 2, 3]")


def test_ensure_dict_rejects_unsupported_type():
    service, _ = _make_service()
    with pytest.raises(StateStoreError):
        service._ensure_dict(42)


def test_validate_model_returns_payload_unchanged_when_no_model_configured():
    service, _ = _make_service()
    payload = {"name": "a", "count": 1}
    assert service._validate_model(payload) is payload


def test_validate_model_dumps_to_dict_by_default():
    service, _ = _make_service(model=Widget)
    result = service._validate_model({"name": "a", "count": 1})
    assert result == {"name": "a", "count": 1}


def test_validate_model_returns_instance_when_requested():
    service, _ = _make_service(model=Widget)
    result = service._validate_model({"name": "a", "count": 1}, return_model=True)
    assert isinstance(result, Widget)
    assert result.count == 1


def test_validate_model_raises_on_invalid_payload():
    service, _ = _make_service(model=Widget)
    with pytest.raises(StateStoreError):
        service._validate_model({"count": "not-a-number"})


def test_load_returns_default_when_key_missing():
    service, store = _make_service()
    store.get_state.return_value = _state_response(None)
    result = service.load(key="k", default={"count": 0})
    assert result == {"count": 0}


def test_load_default_dict_is_copied_not_shared():
    service, store = _make_service()
    store.get_state.return_value = _state_response(None)
    shared_default = {"count": 0}
    result = service.load(key="k", default=shared_default)
    result["count"] = 99
    assert shared_default["count"] == 0


def test_load_parses_and_returns_existing_state():
    service, store = _make_service()
    store.get_state.return_value = _state_response({"count": 5})
    assert service.load(key="k") == {"count": 5}


def test_load_raises_when_state_is_not_a_dict():
    service, store = _make_service()
    store.get_state.return_value = _state_response([1, 2])
    with pytest.raises(StateStoreError):
        service.load(key="k")


def test_load_wraps_backend_errors():
    service, store = _make_service()
    store.get_state.side_effect = RuntimeError("boom")
    with pytest.raises(StateStoreError):
        service.load(key="k")


def test_load_with_etag_returns_etag_for_existing_key():
    service, store = _make_service()
    store.get_state.return_value = _state_response({"count": 1}, etag="v1")
    payload, etag = service.load_with_etag(key="k")
    assert payload == {"count": 1}
    assert etag == "v1"


def test_load_with_etag_returns_none_for_missing_key():
    service, store = _make_service()
    store.get_state.return_value = _state_response(None)
    payload, etag = service.load_with_etag(key="k", default={})
    assert payload == {}
    assert etag is None


def test_load_many_strips_prefix_and_skips_empty_items():
    service, store = _make_service(key_prefix="blog:")
    store.get_bulk_state.return_value = [
        _bulk_item("blog:a", {"count": 1}),
        _bulk_item("blog:b", None),
    ]
    result = service.load_many(["a", "b"])
    assert result == {"a": {"count": 1}}


def test_save_qualifies_key_and_serializes_payload():
    service, store = _make_service(key_prefix="blog:")
    service.save(key="post-1", value={"count": 1})
    _, kwargs = store.save_state.call_args
    args = store.save_state.call_args.args
    assert args[0] == "blog:post-1"
    assert json.loads(args[1]) == {"count": 1}


def test_save_sets_ttl_metadata():
    service, store = _make_service()
    service.save(key="k", value={}, ttl_in_seconds=60)
    metadata = store.save_state.call_args.kwargs["state_metadata"]
    assert metadata["ttlInSeconds"] == "60"


def test_save_does_not_override_explicit_ttl_metadata():
    service, store = _make_service()
    service.save(
        key="k", value={}, ttl_in_seconds=60, state_metadata={"ttlInSeconds": "10"}
    )
    metadata = store.save_state.call_args.kwargs["state_metadata"]
    assert metadata["ttlInSeconds"] == "10"


def test_save_mirrors_to_disk_when_enabled(tmp_path):
    service, store = _make_service(mirror_writes=True, local_mirror_path=str(tmp_path))
    service.save(key="post-1", value={"count": 1})
    mirrored = json.loads((tmp_path / "post-1.json").read_text())
    assert mirrored == {"count": 1}


def test_save_mirror_merges_with_existing_file(tmp_path):
    service, store = _make_service(mirror_writes=True, local_mirror_path=str(tmp_path))
    service.save(key="post-1", value={"a": 1})
    service.save(key="post-1", value={"b": 2})
    mirrored = json.loads((tmp_path / "post-1.json").read_text())
    assert mirrored == {"a": 1, "b": 2}


def test_delete_calls_underlying_store():
    service, store = _make_service()
    service.delete(key="k", etag="v1")
    store.delete_state.assert_called_once()
    assert store.delete_state.call_args.kwargs["etag"] == "v1"


def test_delete_wraps_backend_errors():
    service, store = _make_service()
    store.delete_state.side_effect = RuntimeError("boom")
    with pytest.raises(StateStoreError):
        service.delete(key="k")


def test_exists_true_when_etag_present():
    service, store = _make_service()
    store.get_state.return_value = _state_response({}, etag="v1")
    assert service.exists(key="k") is True


def test_exists_false_when_key_missing():
    service, store = _make_service()
    store.get_state.return_value = _state_response(None)
    assert service.exists(key="k") is False


def test_save_many_saves_each_item_with_qualified_key():
    service, store = _make_service(key_prefix="blog:")
    service.save_many({"a": {"count": 1}, "b": {"count": 2}})
    calls = {c.args[0]: json.loads(c.args[1]) for c in store.save_state.call_args_list}
    assert calls == {"blog:a": {"count": 1}, "blog:b": {"count": 2}}


def test_execute_transaction_passes_operations_and_metadata():
    service, store = _make_service()
    ops = [{"operation": "upsert", "request": {"key": "k"}}]
    service.execute_transaction(ops, metadata={"m": "1"})
    store.execute_state_transaction.assert_called_once_with(ops, metadata={"m": "1"})


def test_with_retries_retries_until_success():
    service, _ = _make_service(retry_attempts=3)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert service._with_retries(flaky) == "ok"
    assert calls["n"] == 3


def test_with_retries_raises_after_exhausting_attempts():
    service, _ = _make_service(retry_attempts=2)

    def always_fails():
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        service._with_retries(always_fails)


def test_deep_merge_overrides_scalars_and_merges_nested_dicts():
    original = {"a": 1, "nested": {"x": 1, "y": 2}}
    updates = {"a": 2, "nested": {"y": 3, "z": 4}}
    assert _deep_merge(original, updates) == {
        "a": 2,
        "nested": {"x": 1, "y": 3, "z": 4},
    }


def test_deep_merge_does_not_mutate_original():
    original = {"nested": {"x": 1}}
    _deep_merge(original, {"nested": {"y": 2}})
    assert original == {"nested": {"x": 1}}
