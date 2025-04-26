import os

from dapr_agents.types.llm import AudioSpeechRequest
from dapr_agents import OpenAIAudioClient
from dotenv import load_dotenv

load_dotenv()
client = OpenAIAudioClient()

# Define the text to convert to speech
text_to_speech = (
    "Dapr Agents is an open-source framework for researchers and developers"
)

# Create a request for TTS
tts_request = AudioSpeechRequest(
    model="tts-1", input=text_to_speech, voice="fable", response_format="mp3"
)

# Generate the audio - returns a byte string
audio_bytes = client.create_speech(request=tts_request)

# You can also automatically create the audio file by passing the file name as an argument
# client.create_speech(request=tts_request, file_name=output_path)

# Save the audio to an MP3 file
output_path = "output_speech.mp3"
with open(output_path, "wb") as audio_file:
    audio_file.write(audio_bytes)

print(f"Audio saved to {output_path}")

os.remove(output_path)
print(f"File {output_path} has been deleted.")
