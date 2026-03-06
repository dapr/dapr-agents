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

from typing import Union
import httpx


class HTTPHelper:
    """
    HTTP operations helper.
    """

    @staticmethod
    def configure_timeout(timeout: Union[int, float, dict]) -> httpx.Timeout:
        """
        Configure the timeout setting for the HTTP client.
        :param timeout: Timeout in seconds or a dictionary of timeout configurations.
        :return: An httpx.Timeout instance configured with the provided timeout.
        """
        if isinstance(timeout, (int, float)):
            return httpx.Timeout(timeout)
        elif isinstance(timeout, dict):
            return httpx.Timeout(**timeout)
        else:
            return httpx.Timeout(30)
