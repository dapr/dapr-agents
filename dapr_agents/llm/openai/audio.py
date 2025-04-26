from dapr_agents.llm.openai.client.base import OpenAIClientBase
from dapr_agents.llm.utils import RequestHandler
from dapr_agents.types.llm import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse,
)
from typing import Union, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OpenAIAudioClient(OpenAIClientBase):
    """
    Client for handling OpenAI's audio functionalities, including speech generation, transcription, and translation.
    Inherits shared logic and configuration from OpenAIClientBase.
    """

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the private attributes specific to the audio client.
        """
        self._api = "audio"
        super().model_post_init(__context)

    def create_speech(
        self,
        request: Union[AudioSpeechRequest, Dict[str, Any]],
        file_name: Optional[str] = None,
    ) -> Union[bytes, None]:
        """
        Generate speech audio from text and optionally save it to a file.

        Args:
            request (Union[AudioSpeechRequest, Dict[str, Any]]): The request parameters for speech generation.
            file_name (Optional[str]): Optional file name to save the generated audio.

        Returns:
            Union[bytes, None]: The generated audio content as bytes if no file_name is provided, otherwise None.
        """
        # Transform dictionary to Pydantic object if needed
        validated_request: AudioSpeechRequest = RequestHandler.validate_request(
            request, AudioSpeechRequest
        )

        logger.info(f"Using model '{validated_request.model}' for speech generation.")

        input_text = validated_request.input

        max_chunk_size = 4096

        if len(input_text) > max_chunk_size:
            logger.info(
                f"Input exceeds {max_chunk_size} characters. Splitting into smaller chunks."
            )

        # Split input text into manageable chunks
        def split_text(text, max_size):
            chunks = []
            while len(text) > max_size:
                split_index = text.rfind(". ", 0, max_size) + 1 or max_size
                chunks.append(text[:split_index].strip())
                text = text[split_index:].strip()
            chunks.append(text)
            return chunks

        text_chunks = split_text(input_text, max_chunk_size)

        audio_chunks = []

        try:
            for chunk in text_chunks:
                validated_request.input = chunk
                with self.client.with_streaming_response.audio.speech.create(
                    **validated_request.model_dump()
                ) as response:
                    if file_name:
                        # Write each chunk incrementally to the file
                        logger.info(f"Saving audio chunk to file: {file_name}")
                        with open(file_name, "ab") as audio_file:
                            for chunk in response.iter_bytes():
                                audio_file.write(chunk)
                    else:
                        # Collect all chunks in memory for combining
                        audio_chunks.extend(response.iter_bytes())

            if file_name:
                return None
            else:
                # Combine all chunks into one bytes object
                return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"Failed to create or save speech: {e}")
            raise ValueError(f"An error occurred during speech generation: {e}")

    def create_transcription(
        self, request: Union[AudioTranscriptionRequest, Dict[str, Any]]
    ) -> AudioTranscriptionResponse:
        """
        Transcribe audio to text.

        Args:
            request (Union[AudioTranscriptionRequest, Dict[str, Any]]): The request parameters for transcription.

        Returns:
            AudioTranscriptionResponse: The transcription result.
        """
        validated_request: AudioTranscriptionRequest = RequestHandler.validate_request(
            request, AudioTranscriptionRequest
        )

        logger.info(f"Using model '{validated_request.model}' for transcription.")

        response = self.client.audio.transcriptions.create(
            file=validated_request.file,
            **validated_request.model_dump(exclude={"file"}),
        )
        return response

    def create_translation(
        self, request: Union[AudioTranslationRequest, Dict[str, Any]]
    ) -> AudioTranslationResponse:
        """
        Translate audio to English.

        Args:
            request (Union[AudioTranslationRequest, Dict[str, Any]]): The request parameters for translation.

        Returns:
            AudioTranslationResponse: The translation result.
        """
        validated_request: AudioTranslationRequest = RequestHandler.validate_request(
            request, AudioTranslationRequest
        )

        logger.info(f"Using model '{validated_request.model}' for translation.")

        response = self.client.audio.translations.create(
            file=validated_request.file,
            **validated_request.model_dump(exclude={"file"}),
        )
        return response
