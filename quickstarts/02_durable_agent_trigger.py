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

from dapr_agents import trigger_agent


def main() -> None:
    result = trigger_agent(
        "WeatherAgent",
        input={"task": "What is the weather in London?"},
        app_id="weather-agent",
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
