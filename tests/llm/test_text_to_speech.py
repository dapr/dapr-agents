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

import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def test_text():
    """Sample text for TTS tests."""
    return "Test speech for ElevenLabs"


@pytest.fixture
def fake_audio_bytes():
    """Fake audio bytes for mocking."""
    return b"fake-audio-bytes"


def generate_speech(text, output_path):
    """Generate speech audio and save to file."""
    from dapr_agents import ElevenLabsSpeechClient

    client = ElevenLabsSpeechClient(
        model="eleven_multilingual_v2",
        voice="JBFqnCBsd6RMkjVDRZzb",
    )
    audio_bytes = client.create_speech(
        text=text,
        output_format="mp3_44100_128",
    )
    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_bytes)
    return output_path


@patch("dapr_agents.ElevenLabsSpeechClient")
def test_generate_speech(mock_client_class, tmp_path, test_text, fake_audio_bytes):
    """Test speech generation and file output with mocked ElevenLabs client."""
    output_path = tmp_path / "test_output.mp3"
    mock_client = MagicMock()
    mock_client.create_speech.return_value = fake_audio_bytes
    mock_client_class.return_value = mock_client

    result_path = generate_speech(test_text, str(output_path))
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0
    with open(result_path, "rb") as f:
        data = f.read()
        assert data == fake_audio_bytes
    os.remove(result_path)
