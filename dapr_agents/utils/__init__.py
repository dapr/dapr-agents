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

from .semver import is_version_supported  # re-export for convenience

from .signal_handlers import add_signal_handlers_cross_platform
from .signal_mixin import SignalHandlingMixin

__all__ = ["add_signal_handlers_cross_platform", "SignalHandlingMixin"]
