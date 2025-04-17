from dapr_agents.llm.elevenlabs.client import ElevenLabsClientBase
from typing import Optional, Union, Any
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class ElevenLabsSpeechClient(ElevenLabsClientBase):
    """
    Client for ElevenLabs speech generation functionality.
    Handles text-to-speech conversions with customizable options.
    """

    voice: Optional[Any] = Field(
        default=None,
        description="Default voice (ID, name, or object) for speech generation.",
    )
    model: Optional[str] = Field(
        default="eleven_multilingual_v2",
        description="Default model for speech generation.",
    )
    output_format: Optional[str] = Field(
        default="mp3_44100_128", description="Default audio output format."
    )
    optimize_streaming_latency: Optional[int] = Field(
        default=0,
        description="Default latency optimization level (0 means no optimizations).",
    )
    voice_settings: Optional[Any] = Field(
        default=None,
        description="Default voice settings (stability, similarity boost, etc.).",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization logic for the ElevenLabsSpeechClient.
        Dynamically imports ElevenLabs components and validates voice attributes.
        """
        super().model_post_init(__context)

        from elevenlabs import VoiceSettings
        from elevenlabs.client import DEFAULT_VOICE

        # Set default voice settings if not already set
        if self.voice_settings is None:
            self.voice_settings = VoiceSettings(
                stability=DEFAULT_VOICE.settings.stability,
                similarity_boost=DEFAULT_VOICE.settings.similarity_boost,
                style=DEFAULT_VOICE.settings.style,
                use_speaker_boost=DEFAULT_VOICE.settings.use_speaker_boost,
            )

        # Set default voice if not provided
        if self.voice is None:
            self.voice = DEFAULT_VOICE

    def create_speech(
        self,
        text: str,
        file_name: Optional[str] = None,
        voice: Optional[Union[str, Any]] = None,
        model: Optional[str] = None,
        output_format: Optional[str] = None,
        optimize_streaming_latency: Optional[int] = None,
        voice_settings: Optional[Any] = None,
        overwrite_file: bool = True,
    ) -> Union[bytes, None]:
        """
        Generate speech audio from text and optionally save it to a file.

        Args:
            text (str): The text to convert to speech.
            file_name (Optional[str]): Optional file name to save the generated audio.
            voice (Optional[Union[str, Voice]]): Override default voice for this request (ID, name, or object).
            model (Optional[str]): Override default model for this request.
            output_format (Optional[str]): Override default output format for this request.
            optimize_streaming_latency (Optional[int]): Override default latency optimization level.
            voice_settings (Optional[VoiceSettings]): Override default voice settings (stability, similarity boost, etc.).
            overwrite_file (bool): Whether to overwrite the file if it exists. Defaults to True.

        Returns:
            Union[bytes, None]: The generated audio as bytes if no `file_name` is provided; otherwise, None.
        """
        # Apply defaults if arguments are not provided
        voice = voice or self.voice
        model = model or self.model
        output_format = output_format or self.output_format
        optimize_streaming_latency = (
            optimize_streaming_latency or self.optimize_streaming_latency
        )
        voice_settings = voice_settings or self.voice_settings

        logger.info(f"Generating speech with voice '{voice}', model '{model}'.")

        try:
            audio_chunks = self.client.generate(
                text=text,
                voice=voice,
                model=model,
                output_format=output_format,
                optimize_streaming_latency=optimize_streaming_latency,
                voice_settings=voice_settings,
            )

            if file_name:
                file_mode = "wb" if overwrite_file else "ab"
                logger.info(f"Saving audio to file: {file_name} (mode: {file_mode})")
                with open(file_name, file_mode) as audio_file:
                    for chunk in audio_chunks:
                        audio_file.write(chunk)
                logger.info(f"Audio saved to {file_name}")
                return None
            else:
                logger.info("Collecting audio bytes.")
                return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise ValueError(f"An error occurred during speech generation: {e}")
