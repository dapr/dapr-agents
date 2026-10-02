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

"""Rate-limited WARNING logging for conditions that can repeat per message."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Hashable

logger = logging.getLogger(__name__)

DEFAULT_WARNING_INTERVAL_SECONDS = 60.0


class WarningThrottle:
    """Logs at most one WARNING per key per interval.

    Warnings inside the interval are counted, and the next one that is logged
    says how many were suppressed. Keys should come from a small, fixed set
    (a pool, a route), since one timestamp is kept per key.
    """

    def __init__(
        self,
        interval_seconds: float = DEFAULT_WARNING_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = interval_seconds
        self._clock = clock
        self._last: dict[Hashable, float] = {}
        self._suppressed: dict[Hashable, int] = {}
        self._lock = threading.Lock()

    def warn(
        self, log: logging.Logger, key: Hashable, message: str, *args: Any
    ) -> bool:
        """Log ``message % args`` at WARNING unless ``key`` logged recently.

        Returns:
            True when the warning was logged, False when it was suppressed.
        """
        with self._lock:
            now = self._clock()
            last = self._last.get(key)
            if last is not None and now - last < self._interval:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return False
            self._last[key] = now
            suppressed = self._suppressed.pop(key, 0)
        if suppressed:
            message += " (%d similar warnings suppressed in the last %gs)"
            args = (*args, suppressed, self._interval)
        log.warning(message, *args)
        return True


__all__ = ["DEFAULT_WARNING_INTERVAL_SECONDS", "WarningThrottle"]
