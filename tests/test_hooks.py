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

from dapr_agents.hooks import Deny


def test_deny_backwards_compat_reason_only():
    d = Deny(reason="blocked")
    assert d.reason == "blocked"
    assert d.code is None
    assert d.details is None


def test_deny_with_code_and_details():
    d = Deny(
        reason="JWT failed",
        code="oauth.invalid_signature",
        details={"issuer_attempted": "https://acme.us.auth0.com/"},
    )
    assert d.code == "oauth.invalid_signature"
    assert d.details["issuer_attempted"] == "https://acme.us.auth0.com/"


def test_deny_defaults_to_all_none():
    d = Deny()
    assert d.reason is None
    assert d.code is None
    assert d.details is None


def test_deny_code_accepts_empty_string():
    d = Deny(reason="blocked", code="")
    assert d.code == ""


def test_deny_details_accepts_arbitrary_nested_dict():
    d = Deny(details={"nested": {"deep": [1, 2, 3]}})
    assert d.details["nested"]["deep"] == [1, 2, 3]


def test_deny_equality():
    a = Deny(reason="x", code="y", details={"k": 1})
    b = Deny(reason="x", code="y", details={"k": 1})
    assert a == b
