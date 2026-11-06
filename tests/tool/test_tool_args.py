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

"""Tests for how AgentTool passes validated arguments to the tool function."""

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

from dapr_agents.tool import tool


class Flight(BaseModel):
    origin: str
    destination: str


@dataclass
class Seat:
    row: int
    letter: str


@tool
def book_flight(flight: Flight, seats: List[Seat], note: Optional[str] = "none"):
    """Book a flight."""
    return flight, seats, note


FLIGHT = {"origin": "BEY", "destination": "CDG"}
SEATS = [{"row": 12, "letter": "A"}]


def test_run_passes_declared_types_to_function():
    flight, seats, note = book_flight.run(flight=FLIGHT, seats=SEATS)
    assert flight == Flight(origin="BEY", destination="CDG")
    assert seats == [Seat(row=12, letter="A")]
    assert note == "none"


async def test_arun_passes_declared_types_to_function():
    flight, seats, _ = await book_flight.arun(flight=FLIGHT, seats=SEATS)
    assert isinstance(flight, Flight)
    assert isinstance(seats[0], Seat)


def test_none_argument_falls_back_to_default():
    _, _, note = book_flight.run(flight=FLIGHT, seats=SEATS, note=None)
    assert note == "none"


class BookFlightArgs(BaseModel):
    flight: Flight


def test_explicit_args_model_still_passes_plain_data():
    @tool(args_model=BookFlightArgs)
    def book(flight):
        """Book a flight."""
        return flight

    assert book.run(flight=FLIGHT) == FLIGHT
