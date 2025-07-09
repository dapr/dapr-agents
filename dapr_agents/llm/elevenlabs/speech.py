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

    voice: Optional[str] = Field(
        default="JBFqnCBsd6RMkjVDRZzb",  # George
        description="Default voice (ID, name) for speech generation.",
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

        if self.voice_settings is None:
            self.voice_settings = self.client.voices.settings.get_default()

    def create_speech(
        self,
        text: str,
        file_name: Optional[str] = None,
        voice: Optional[Union[str, Any]] = None,
        model: Optional[str] = None,
        output_format: Optional[str] = None,
        optimize_streaming_latency: Optional[int] = None,
        voice_settings: Optional[Any] = None,
        pronunciation_dictionary_locators: Optional[Any] = None,
        seed: Optional[int] = None,
        previous_text: Optional[str] = None,
        next_text: Optional[str] = None,
        previous_request_ids: Optional[Any] = None,
        next_request_ids: Optional[Any] = None,
        language_code: Optional[str] = None,
        use_pvc_as_ivc: Optional[bool] = None,
        apply_text_normalization: Optional[Any] = None,
        apply_language_text_normalization: Optional[bool] = None,
        enable_logging: Optional[bool] = None,
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
            pronunciation_dictionary_locators (Optional[Any]): Pronunciation dictionary locators for custom pronunciations.
            seed (Optional[int]): Seed for deterministic output.
            previous_text (Optional[str]): Text before this request for continuity.
            next_text (Optional[str]): Text after this request for continuity.
            previous_request_ids (Optional[Any]): Previous request IDs for continuity.
            next_request_ids (Optional[Any]): Next request IDs for continuity.
            language_code (Optional[str]): Enforce a specific language code.
            use_pvc_as_ivc (Optional[bool]): Use IVC version of the voice for lower latency.
            apply_text_normalization (Optional[Any]): Control text normalization ('auto', 'on', 'off').
            apply_language_text_normalization (Optional[bool]): Language-specific normalization.
            enable_logging (Optional[bool]): Enable/disable logging for privacy.
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
            audio_chunks = self.client.text_to_speech.convert(
                voice_id=voice,
                text=text,
                model_id=model,
                output_format=output_format,
                optimize_streaming_latency=optimize_streaming_latency,
                voice_settings=voice_settings,
                pronunciation_dictionary_locators=pronunciation_dictionary_locators,
                seed=seed,
                previous_text=previous_text,
                next_text=next_text,
                previous_request_ids=previous_request_ids,
                next_request_ids=next_request_ids,
                language_code=language_code,
                use_pvc_as_ivc=use_pvc_as_ivc,
                apply_text_normalization=apply_text_normalization,
                apply_language_text_normalization=apply_language_text_normalization,
                enable_logging=enable_logging,
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
