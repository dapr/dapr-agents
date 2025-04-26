from dapr_agents.document.reader.pdf.pypdf import PyPDFReader
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents import WorkflowApp
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
from typing import Dict, Any, List
from pydantic import BaseModel
from pathlib import Path
from dapr_agents import OpenAIAudioClient
from dapr_agents.types.llm import AudioSpeechRequest
from pydub import AudioSegment
import io
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the WorkflowApp
wfapp = WorkflowApp()


# Define structured output models
class SpeakerEntry(BaseModel):
    name: str
    text: str


class PodcastDialogue(BaseModel):
    participants: List[SpeakerEntry]


# Define Workflow logic
@wfapp.workflow(name="doc2podcast")
def doc2podcast(ctx: DaprWorkflowContext, input: Dict[str, Any]):
    # Extract pre-validated input
    podcast_name = input["podcast_name"]
    host_config = input["host"]
    participant_configs = input["participants"]
    max_rounds = input["max_rounds"]
    file_input = input["pdf_url"]
    output_transcript_path = input["output_transcript_path"]
    output_audio_path = input["output_audio_path"]
    audio_model = input["audio_model"]

    # Step 1: Assign voices to the team
    team_config = yield ctx.call_activity(
        assign_podcast_voices,
        input={
            "host_config": host_config,
            "participant_configs": participant_configs,
        },
    )

    # Step 2: Read PDF and get documents
    file_path = yield ctx.call_activity(download_pdf, input=file_input)
    documents = yield ctx.call_activity(read_pdf, input={"file_path": file_path})

    # Step 3: Initialize context and transcript parts
    accumulated_context = ""
    transcript_parts = []
    total_iterations = len(documents)

    for chunk_index, document in enumerate(documents):
        # Generate the intermediate prompt
        document_with_context = {
            "text": document["text"],
            "iteration_index": chunk_index + 1,
            "total_iterations": total_iterations,
            "context": accumulated_context,
            "participants": [p["name"] for p in team_config["participants"]],
        }
        generated_prompt = yield ctx.call_activity(
            generate_prompt, input=document_with_context
        )

        # Use the prompt to generate the structured dialogue
        prompt_parameters = {
            "podcast_name": podcast_name,
            "host_name": team_config["host"]["name"],
            "prompt": generated_prompt,
            "max_rounds": max_rounds,
        }
        dialogue_entry = yield ctx.call_activity(
            generate_transcript, input=prompt_parameters
        )

        # Update context and transcript parts
        conversations = dialogue_entry["participants"]
        for participant in conversations:
            accumulated_context += f" {participant['name']}: {participant['text']}"
            transcript_parts.append(participant)

    # Step 4: Write the final transcript to a file
    yield ctx.call_activity(
        write_transcript_to_file,
        input={
            "podcast_dialogue": transcript_parts,
            "output_path": output_transcript_path,
        },
    )

    # Step 5: Convert transcript to audio using team_config
    yield ctx.call_activity(
        convert_transcript_to_audio,
        input={
            "transcript_parts": transcript_parts,
            "output_path": output_audio_path,
            "voices": team_config,
            "model": audio_model,
        },
    )


@wfapp.task
def assign_podcast_voices(
    host_config: Dict[str, Any], participant_configs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Assign voices to the podcast host and participants.

    Args:
        host_config: Dictionary containing the host's configuration (name and optionally a voice).
        participant_configs: List of dictionaries containing participants' configurations (name and optionally a voice).

    Returns:
        A dictionary with the updated `host` and `participants`, including their assigned voices.
    """
    allowed_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    assigned_voices = set()  # Track assigned voices to avoid duplication

    # Assign voice to the host if not already specified
    if "voice" not in host_config:
        host_config["voice"] = next(
            voice for voice in allowed_voices if voice not in assigned_voices
        )
    assigned_voices.add(host_config["voice"])

    # Assign voices to participants, ensuring no duplicates
    updated_participants = []
    for participant in participant_configs:
        if "voice" not in participant:
            participant["voice"] = next(
                voice for voice in allowed_voices if voice not in assigned_voices
            )
        assigned_voices.add(participant["voice"])
        updated_participants.append(participant)

    # Return the updated host and participants
    return {
        "host": host_config,
        "participants": updated_participants,
    }


@wfapp.task
def download_pdf(pdf_url: str, local_directory: str = ".") -> str:
    """
    Downloads a PDF file from a URL and saves it locally, automatically determining the filename.
    """
    try:
        parsed_url = urlparse(pdf_url)
        filename = unquote(Path(parsed_url.path).name)

        if not filename:
            raise ValueError("Invalid URL: Cannot determine filename from the URL.")

        filename = filename.replace(" ", "_")
        local_directory_path = Path(local_directory).resolve()
        local_directory_path.mkdir(parents=True, exist_ok=True)
        local_file_path = local_directory_path / filename

        if not local_file_path.exists():
            logger.info(f"Downloading PDF from {pdf_url}...")
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(local_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            logger.info(f"PDF saved to {local_file_path}")
        else:
            logger.info(f"PDF already exists at {local_file_path}")

        return str(local_file_path)
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        raise


@wfapp.task
def read_pdf(file_path: str) -> List[dict]:
    """
    Reads and extracts text from a PDF document.
    """
    try:
        reader = PyPDFReader()
        documents = reader.load(file_path)
        return [doc.model_dump() for doc in documents]
    except Exception as e:
        logger.error(f"Error reading document: {e}")
        raise


@wfapp.task
def generate_prompt(
    text: str,
    iteration_index: int,
    total_iterations: int,
    context: str,
    participants: List[str],
) -> str:
    """
    Generate a prompt dynamically for the chunk.
    """
    logger.info(f"Processing iteration {iteration_index} of {total_iterations}.")
    instructions = f"""
    CONTEXT:
    - Previous conversation: {context.strip() or "No prior context available."}
    - This is iteration {iteration_index} of {total_iterations}.
    """

    if participants:
        participant_names = ", ".join(participants)
        instructions += f"\nPARTICIPANTS: {participant_names}"
    else:
        instructions += "\nPARTICIPANTS: None (Host-only conversation)"

    if iteration_index == 1:
        instructions += """
        INSTRUCTIONS:
        - Begin with a warm welcome to the podcast titled 'Podcast Name'.
        - Introduce the host and the participants (if available).
        - Provide an overview of the topics to be discussed in this episode.
        """
    elif iteration_index == total_iterations:
        instructions += """
        INSTRUCTIONS:
        - Conclude the conversation with a summary of the discussion.
        - Include farewell messages from the host and participants.
        """
    else:
        instructions += """
        INSTRUCTIONS:
        - Continue the conversation smoothly without re-introducing the podcast.
        - Follow up on the previous discussion points and introduce the next topic naturally.
        """

    instructions += """
    TASK:
    - Use the provided TEXT to guide this part of the conversation.
    - Alternate between speakers, ensuring a natural conversational flow.
    - Keep responses concise and aligned with the context.
    """
    return f"{instructions}\nTEXT:\n{text.strip()}"


@wfapp.task(
    """
    Generate a structured podcast dialogue based on the context and text provided.
    The podcast is titled '{podcast_name}' and is hosted by {host_name}.
    If participants are available, each speaker is limited to a maximum of {max_rounds} turns per iteration.
    A "round" is defined as one turn by the host followed by one turn by a participant.
    The podcast should alternate between the host and participants.
    If participants are not available, the host drives the conversation alone.
    Keep the dialogue concise and ensure a natural conversational flow.
    {prompt}
"""
)
def generate_transcript(
    podcast_name: str, host_name: str, prompt: str, max_rounds: int
) -> PodcastDialogue:
    pass


@wfapp.task
def write_transcript_to_file(
    podcast_dialogue: List[Dict[str, Any]], output_path: str
) -> None:
    """
    Write the final structured transcript to a file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            import json

            json.dump(podcast_dialogue, file, ensure_ascii=False, indent=4)
        logger.info(f"Podcast dialogue successfully written to {output_path}")
    except Exception as e:
        logger.error(f"Error writing podcast dialogue to file: {e}")
        raise


@wfapp.task
def convert_transcript_to_audio(
    transcript_parts: List[Dict[str, Any]],
    output_path: str,
    voices: Dict[str, Any],
    model: str = "tts-1",
) -> None:
    """
    Converts a transcript into a single audio file using the OpenAI Audio Client and pydub for concatenation.

    Args:
        transcript_parts: List of dictionaries containing speaker and text.
        output_path: File path to save the final audio.
        voices: Dictionary containing "host" and "participants" with their assigned voices.
        model: TTS model to use (default: "tts-1").
    """
    try:
        client = OpenAIAudioClient()
        combined_audio = AudioSegment.silent(duration=500)  # Start with a short silence

        # Build voice mapping
        voice_mapping = {voices["host"]["name"]: voices["host"]["voice"]}
        voice_mapping.update({p["name"]: p["voice"] for p in voices["participants"]})

        for part in transcript_parts:
            speaker_name = part["name"]
            speaker_text = part["text"]
            assigned_voice = voice_mapping.get(
                speaker_name, "alloy"
            )  # Default to "alloy" if not found

            # Log assigned voice for debugging
            logger.info(
                f"Generating audio for {speaker_name} using voice '{assigned_voice}'."
            )

            # Create TTS request
            tts_request = AudioSpeechRequest(
                model=model,
                input=speaker_text,
                voice=assigned_voice,
                response_format="mp3",
            )

            # Generate the audio
            audio_bytes = client.create_speech(request=tts_request)

            # Create an AudioSegment from the audio bytes
            audio_chunk = AudioSegment.from_file(
                io.BytesIO(audio_bytes), format=tts_request.response_format
            )

            # Append the audio to the combined segment
            combined_audio += audio_chunk + AudioSegment.silent(duration=300)

        # Export the combined audio to the output file
        combined_audio.export(output_path, format="mp3")
        logger.info(f"Podcast audio successfully saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during audio generation: {e}")
        raise


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    def load_config(file_path: str) -> dict:
        """Load configuration from a JSON or YAML file."""
        with open(file_path, "r") as file:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                return yaml.safe_load(file)
            elif file_path.endswith(".json"):
                return json.load(file)
            else:
                raise ValueError("Unsupported file format. Use JSON or YAML.")

    # CLI Argument Parser
    parser = argparse.ArgumentParser(description="Document to Podcast Workflow")
    parser.add_argument("--config", type=str, help="Path to a JSON/YAML config file.")
    parser.add_argument("--pdf_url", type=str, help="URL of the PDF document.")
    parser.add_argument("--podcast_name", type=str, help="Name of the podcast.")
    parser.add_argument("--host_name", type=str, help="Name of the host.")
    parser.add_argument("--host_voice", type=str, help="Voice for the host.")
    parser.add_argument(
        "--participants", type=str, nargs="+", help="List of participant names."
    )
    parser.add_argument(
        "--max_rounds", type=int, default=4, help="Number of turns per round."
    )
    parser.add_argument(
        "--output_transcript_path", type=str, help="Path to save the output transcript."
    )
    parser.add_argument(
        "--output_audio_path", type=str, help="Path to save the final audio file."
    )
    parser.add_argument(
        "--audio_model", type=str, default="tts-1", help="Audio model for TTS."
    )

    args = parser.parse_args()

    # Load config file if provided
    config = load_config(args.config) if args.config else {}

    # Merge CLI and Config inputs
    user_input = {
        "pdf_url": args.pdf_url or config.get("pdf_url"),
        "podcast_name": args.podcast_name
        or config.get("podcast_name", "Default Podcast"),
        "host": {
            "name": args.host_name or config.get("host", {}).get("name", "Host"),
            "voice": args.host_voice or config.get("host", {}).get("voice", "alloy"),
        },
        "participants": config.get("participants", []),
        "max_rounds": args.max_rounds or config.get("max_rounds", 4),
        "output_transcript_path": args.output_transcript_path
        or config.get("output_transcript_path", "podcast_dialogue.json"),
        "output_audio_path": args.output_audio_path
        or config.get("output_audio_path", "final_podcast.mp3"),
        "audio_model": args.audio_model or config.get("audio_model", "tts-1"),
    }

    # Add participants from CLI if provided
    if args.participants:
        user_input["participants"].extend({"name": name} for name in args.participants)

    # Validate inputs
    if not user_input["pdf_url"]:
        raise ValueError("PDF URL must be provided via CLI or config file.")

    # Run the workflow
    wfapp.run_and_monitor_workflow_sync(workflow=doc2podcast, input=user_input)
