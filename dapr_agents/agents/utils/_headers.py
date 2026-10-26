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


def parse_header_string(raw_headers: str) -> dict:
    """
    Parse a header string in the form "Key1=Value1,Key2=Value2,..." into a dictionary.

    :param raw_headers: The raw header string.

    :return: A dictionary containing the parsed headers.
    """
    headers: dict[str, str] = {}

    for pair in raw_headers.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            headers[k.strip()] = v.strip()

    return headers
