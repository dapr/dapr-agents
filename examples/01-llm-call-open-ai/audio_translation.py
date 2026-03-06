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

from dapr_agents.types.llm import AudioTranslationRequest
from dapr_agents import OpenAIAudioClient
from dotenv import load_dotenv

load_dotenv()
client = OpenAIAudioClient()

# Specify the audio file to translate
audio_file_path = "speech.mp3"

# Create a translation request
translation_request = AudioTranslationRequest(
    model="whisper-1",
    file=audio_file_path,
    prompt="The user will provide an audio file in Spanish. Translate the audio to English and transcribe the english text, word for word.",
)

# Generate translation
translation_response = client.create_translation(request=translation_request)

# Display the transcription result
if not len(translation_response.text) > 0:
    exit(1)

print("Translation:", translation_response)

words = ["dapr", "agents", "open", "source", "framework", "researchers", "developers"]
normalized_text = translation_response.text.lower()

count = 0
for word in words:
    if word in normalized_text:
        count += 1

if count >= 5:
    print("Success! The transcription contains at least 5 out of 7 words.")
