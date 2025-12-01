from dapr_agents.types.llm import AudioTranscriptionRequest
from dapr_agents import OpenAIAudioClient
from dotenv import load_dotenv

load_dotenv()
client = OpenAIAudioClient()

# Specify the audio file to transcribe
audio_file_path = "speech.mp3"

# Create a transcription request
transcription_request = AudioTranscriptionRequest(
    model="whisper-1", file=audio_file_path
)

############
# You can also use audio bytes:
############
#
# with open(audio_file_path, "rb") as f:
#     audio_bytes = f.read()
#
# transcription_request = AudioTranscriptionRequest(
#     model="whisper-1",
#     file=audio_bytes,  # File as bytes
#     language="en"  # Optional: Specify the language of the audio
# )


# Generate transcription
transcription_response = client.create_transcription(request=transcription_request)

# Display the transcription result
if not len(transcription_response.text) > 0:
    exit(1)

print("Transcription:", transcription_response.text)

words = ["dapr", "agents", "open", "source", "framework", "researchers", "developers"]
normalized_text = transcription_response.text.lower()

count = 0
for word in words:
    if word in normalized_text:
        count += 1

if count >= 5:
    print("Success! The transcription contains at least 5 out of 7 words.")
