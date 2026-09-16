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

from datetime import datetime

import pytest

from dapr_agents.document.fetcher.arxiv import ArxivFetcher


class TestArxivFetcherFormatDate:
    """Unit tests for ArxivFetcher._format_date (no network)."""

    def test_yyyymmdd_string_is_padded_to_twelve_digits(self):
        fetcher = ArxivFetcher()
        assert fetcher._format_date("20240101") == "202401010000"

    def test_yyyymmddhhmm_string_is_returned_unchanged(self):
        fetcher = ArxivFetcher()
        assert fetcher._format_date("202401011200") == "202401011200"

    def test_datetime_object_is_formatted_to_twelve_digits(self):
        fetcher = ArxivFetcher()
        assert fetcher._format_date(datetime(2024, 1, 1, 12, 0)) == "202401011200"

    def test_invalid_string_format_raises(self):
        fetcher = ArxivFetcher()
        with pytest.raises(ValueError, match="Invalid date format"):
            fetcher._format_date("invalid_date")

    def test_invalid_calendar_date_raises(self):
        fetcher = ArxivFetcher()
        with pytest.raises(ValueError, match="Invalid date value"):
            fetcher._format_date("20241332")
