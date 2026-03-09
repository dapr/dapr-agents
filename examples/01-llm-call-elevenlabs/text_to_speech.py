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

from dapr_agents import ElevenLabsSpeechClient
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabsSpeechClient(
    model="eleven_multilingual_v2",  # Default model
    voice="JBFqnCBsd6RMkjVDRZzb",  # 'name': 'George', 'language': 'en', 'labels': {'accent': 'British', 'description': 'warm', 'age': 'middle aged', 'gender': 'male', 'use_case': 'narration'}
)


# Define the text to convert to speech
text = "Dapr Agents is an open-source framework for researchers and developers"

# Create speech from text
audio_bytes = client.create_speech(
    text=text,
    output_format="mp3_44100_128",  # default output format, mp3 with 44.1kHz sample rate at 128kbps.
)

# You can also automatically create the audio file by passing the file name as an argument
# client.create_speech(
#     text=text,
#     output_format="mp3_44100_128", # default output format, mp3 with 44.1kHz sample rate at 128kbps.,
#     file_name='output_speech_auto.mp3'
# )


# Save the audio to an MP3 file
output_path = "output_speech.mp3"
with open(output_path, "wb") as audio_file:
    audio_file.write(audio_bytes)

print(f"Audio saved to {output_path}")

os.remove(output_path)
print(f"File {output_path} has been deleted.")
