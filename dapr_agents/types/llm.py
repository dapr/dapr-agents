from typing import List, Union, Optional, Dict, Any, Literal, IO, Tuple
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from pydantic_core import PydanticUseDefault
from pathlib import Path
from io import BytesIO, BufferedReader


class ElevenLabsClientConfig(BaseModel):
    base_url: Literal[
        "https://api.elevenlabs.io", "https://api.us.elevenlabs.io"
    ] = Field(
        default="https://api.elevenlabs.io",
        description="Base URL for the ElevenLabs API. Defaults to the production environment.",
    )
    api_key: Optional[str] = Field(
        None, description="API key to authenticate with the ElevenLabs API."
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class NVIDIAClientConfig(BaseModel):
    base_url: Optional[str] = Field(
        "https://integrate.api.nvidia.com/v1", description="Base URL for the NVIDIA API"
    )
    api_key: Optional[str] = Field(
        None, description="API key to authenticate the NVIDIA API"
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class DaprInferenceClientConfig:
    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class HFInferenceClientConfig(BaseModel):
    model: Optional[str] = Field(
        None,
        description="Model ID on Hugging Face Hub or URL to a deployed Inference Endpoint. Defaults to a recommended model if not provided.",
    )
    api_key: Optional[Union[str, bool]] = Field(
        None,
        description="Hugging Face API key for authentication. Defaults to the locally saved token. Pass False to skip token.",
    )
    token: Optional[Union[str, bool]] = Field(
        None,
        description="Alias for api_key. Defaults to the locally saved token. Pass False to avoid sending the token.",
    )
    base_url: Optional[str] = Field(
        None,
        description="Base URL to run inference. Cannot be used if model is set. Defaults to None.",
    )
    timeout: Optional[float] = Field(
        None,
        description="Maximum time in seconds to wait for a server response. Defaults to None, meaning it will wait indefinitely.",
    )
    headers: Optional[Dict[str, str]] = Field(
        None,
        description="Additional headers to send to the server. Overrides default headers such as authorization and user-agent.",
    )
    cookies: Optional[Dict[str, str]] = Field(
        None, description="Additional cookies to send with the request."
    )
    proxies: Optional[Any] = Field(None, description="Proxies to use for the request.")

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class OpenAIClientConfig(BaseModel):
    base_url: Optional[str] = Field(None, description="Base URL for the OpenAI API")
    api_key: Optional[str] = Field(
        None, description="API key to authenticate the OpenAI API"
    )
    organization: Optional[str] = Field(
        None, description="Organization name for OpenAI"
    )
    project: Optional[str] = Field(None, description="OpenAI project name.")

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class AzureOpenAIClientConfig(BaseModel):
    api_key: Optional[str] = Field(
        None, description="API key to authenticate the Azure OpenAI API"
    )
    azure_ad_token: Optional[str] = Field(
        None, description="Azure Active Directory token for authentication"
    )
    organization: Optional[str] = Field(
        None, description="Azure organization associated with the OpenAI resource"
    )
    project: Optional[str] = Field(
        None, description="Azure project associated with the OpenAI resource"
    )
    api_version: Optional[str] = Field(
        "2024-07-01-preview", description="API version for Azure OpenAI models"
    )
    azure_endpoint: Optional[str] = Field(
        None, description="Azure endpoint for Azure OpenAI models"
    )
    azure_deployment: Optional[str] = Field(
        default=None, description="Azure deployment for Azure OpenAI models"
    )
    azure_client_id: Optional[str] = Field(
        default=None, description="Client ID for Managed Identity authentication."
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class OpenAIModelConfig(OpenAIClientConfig):
    type: Literal["openai"] = Field(
        "openai", description="Type of the model, must always be 'openai'"
    )
    name: str = Field(default=None, description="Name of the OpenAI model")


class AzureOpenAIModelConfig(AzureOpenAIClientConfig):
    type: Literal["azure_openai"] = Field(
        "azure_openai", description="Type of the model, must always be 'azure_openai'"
    )


class HFHubModelConfig(HFInferenceClientConfig):
    type: Literal["huggingface"] = Field(
        "huggingface", description="Type of the model, must always be 'huggingface'"
    )
    name: str = Field(
        default=None, description="Name of the model available through Hugging Face"
    )


class NVIDIAModelConfig(NVIDIAClientConfig):
    type: Literal["nvidia"] = Field(
        "nvidia", description="Type of the model, must always be 'nvidia'"
    )
    name: str = Field(
        default=None, description="Name of the model available through NVIDIA"
    )


class OpenAIParamsBase(BaseModel):
    """
    Common request settings for OpenAI services.
    """

    model: Optional[str] = Field(None, description="ID of the model to use")
    temperature: Optional[float] = Field(
        0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum number of tokens to generate. Can be None or a positive integer.",
    )
    top_p: Optional[float] = Field(
        1.0, ge=0.0, le=1.0, description="Nucleus sampling probability mass"
    )
    frequency_penalty: Optional[float] = Field(
        0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: Optional[float] = Field(
        0.0, ge=-2.0, le=2.0, description="Presence penalty"
    )
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class OpenAITextCompletionParams(OpenAIParamsBase):
    """
    Specific configs for the text completions endpoint.
    """

    best_of: Optional[int] = Field(
        None, ge=1, description="Number of best completions to generate"
    )
    echo: Optional[bool] = Field(False, description="Whether to echo the prompt")
    logprobs: Optional[int] = Field(
        None, ge=0, le=5, description="Include log probabilities"
    )
    suffix: Optional[str] = Field(None, description="Suffix to append to the prompt")

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class OpenAIChatCompletionParams(OpenAIParamsBase):
    """
    Specific settings for the Chat Completion endpoint.
    """

    logit_bias: Optional[Dict[Union[str, int], float]] = Field(
        None, description="Modify likelihood of specified tokens"
    )
    logprobs: Optional[bool] = Field(
        False, description="Whether to return log probabilities"
    )
    top_logprobs: Optional[int] = Field(
        None, ge=0, le=20, description="Number of top log probabilities to return"
    )
    n: Optional[int] = Field(
        1, ge=1, le=128, description="Number of chat completion choices to generate"
    )
    response_format: Optional[
        Dict[Literal["type"], Literal["text", "json_object"]]
    ] = Field(None, description="Format of the response")
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, max_length=64, description="List of tools the model may call"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Controls which tool is called"
    )
    function_call: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Controls which function is called"
    )
    seed: Optional[int] = Field(None, description="Seed for deterministic sampling")
    user: Optional[str] = Field(
        None, description="Unique identifier representing the end-user"
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class HFHubChatCompletionParams(BaseModel):
    """
    Specific settings for Hugging Face Hub Chat Completion endpoint.
    """

    model: Optional[str] = Field(
        None,
        description="The model to use for chat-completion. Can be a model ID or a URL to a deployed Inference Endpoint.",
    )
    frequency_penalty: Optional[float] = Field(
        0.0,
        description="Penalizes new tokens based on their existing frequency in the text so far.",
    )
    logit_bias: Optional[Dict[Union[str, int], float]] = Field(
        None,
        description="Modify the likelihood of specified tokens appearing in the completion.",
    )
    logprobs: Optional[bool] = Field(
        False,
        description="Whether to return log probabilities of the output tokens or not.",
    )
    max_tokens: Optional[int] = Field(
        100, ge=1, description="Maximum number of tokens allowed in the response."
    )
    n: Optional[int] = Field(None, description="UNUSED. Included for compatibility.")
    presence_penalty: Optional[float] = Field(
        0.0,
        description="Penalizes new tokens based on their presence in the text so far.",
    )
    response_format: Optional[Union[Dict[str, Any], str]] = Field(
        None, description="Grammar constraints. Can be either a JSONSchema or a regex."
    )
    seed: Optional[int] = Field(None, description="Seed for reproducible control flow.")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Up to four strings which trigger the end of the response."
    )
    stream: Optional[bool] = Field(
        False, description="Enable realtime streaming of responses."
    )
    stream_options: Optional[Dict[str, Any]] = Field(
        None, description="Options for streaming completions."
    )
    temperature: Optional[float] = Field(
        1.0, description="Controls randomness of the generations."
    )
    top_logprobs: Optional[int] = Field(
        None, description="Number of most likely tokens to return at each position."
    )
    top_p: Optional[float] = Field(
        None, description="Fraction of the most likely next words to sample from."
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="The tool to use for the completion. Defaults to 'auto'."
    )
    tool_prompt: Optional[str] = Field(
        None, description="A prompt to be appended before the tools."
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="A list of tools the model may call."
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class NVIDIAChatCompletionParams(OpenAIParamsBase):
    """
    Specific settings for the Chat Completion endpoint.
    """

    logit_bias: Optional[Dict[Union[str, int], float]] = Field(
        None, description="Modify likelihood of specified tokens"
    )
    logprobs: Optional[bool] = Field(
        False, description="Whether to return log probabilities"
    )
    top_logprobs: Optional[int] = Field(
        None, ge=0, le=20, description="Number of top log probabilities to return"
    )
    n: Optional[int] = Field(
        1, ge=1, le=128, description="Number of chat completion choices to generate"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, max_length=64, description="List of tools the model may call"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Controls which tool is called"
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class PromptyModelConfig(BaseModel):
    api: Literal["chat", "completion"] = Field(
        "chat", description="The API to use, either 'chat' or 'completion'"
    )
    configuration: Union[
        OpenAIModelConfig, AzureOpenAIModelConfig, HFHubModelConfig, NVIDIAModelConfig
    ] = Field(..., description="Model configuration settings")
    parameters: Union[
        OpenAITextCompletionParams,
        OpenAIChatCompletionParams,
        HFHubChatCompletionParams,
        NVIDIAChatCompletionParams,
    ] = Field(..., description="Parameters for the model request")
    response: Literal["first", "full"] = Field(
        "first",
        description="Determines if full response or just the first one is returned",
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v

    @model_validator(mode="before")
    def sync_model_name(cls, values: dict):
        """
        Ensure that the parameters model name matches the configuration model name.
        """
        configuration = values.get("configuration")
        parameters = values.get("parameters")

        # Ensure the 'configuration' is properly validated as a model, not a dict
        if isinstance(configuration, dict):
            if configuration.get("type") == "openai":
                configuration = OpenAIModelConfig(**configuration)
            elif configuration.get("type") == "azure_openai":
                configuration = AzureOpenAIModelConfig(**configuration)
            elif configuration.get("type") == "huggingface":
                configuration = HFHubModelConfig(**configuration)
            elif configuration.get("type") == "nvidia":
                configuration = NVIDIAModelConfig(**configuration)

        # Ensure 'parameters' is properly validated as a model, not a dict
        if isinstance(parameters, dict):
            if configuration and isinstance(configuration, OpenAIModelConfig):
                parameters = OpenAIChatCompletionParams(**parameters)
            elif configuration and isinstance(configuration, AzureOpenAIModelConfig):
                parameters = OpenAIChatCompletionParams(**parameters)
            elif configuration and isinstance(configuration, HFHubModelConfig):
                parameters = HFHubChatCompletionParams(**parameters)
            elif configuration and isinstance(configuration, NVIDIAModelConfig):
                parameters = NVIDIAChatCompletionParams(**parameters)

        if configuration and parameters:
            # Check if 'name' or 'azure_deployment' is explicitly set
            if "name" in configuration.model_fields_set:
                parameters.model = configuration.name
            elif "azure_deployment" in configuration.model_fields_set:
                parameters.model = configuration.azure_deployment

        values["configuration"] = configuration
        values["parameters"] = parameters
        return values


class PromptyDefinition(BaseModel):
    """Schema for a Prompty definition."""

    name: Optional[str] = Field("", description="Name of the Prompty file.")
    description: Optional[str] = Field(
        "", description="Description of the Prompty file."
    )
    version: Optional[str] = Field("1.0", description="Version of the Prompty.")
    authors: Optional[List[str]] = Field(
        [], description="List of authors for the Prompty."
    )
    tags: Optional[List[str]] = Field([], description="Tags to categorize the Prompty.")
    model: PromptyModelConfig = Field(
        ..., description="Model configuration. Can be either OpenAI or Azure OpenAI."
    )
    inputs: Dict[str, Any] = Field(
        {},
        description="Input parameters for the Prompty. These define the expected inputs.",
    )
    sample: Optional[Union[Dict[str, Any], str]] = Field(
        None,
        description="Sample input or the path to a sample file for testing the Prompty.",
    )
    outputs: Optional[Dict[str, Any]] = Field(
        {},
        description="Optional outputs for the Prompty. Defines expected output format.",
    )
    content: str = Field(
        ..., description="The prompt messages defined in the Prompty file."
    )

    @field_validator("*", mode="before")
    @classmethod
    def none_to_default(cls, v):
        if v is None:
            raise PydanticUseDefault()
        return v


class AudioSpeechRequest(BaseModel):
    model: Optional[Literal["tts-1", "tts-1-hd"]] = Field(
        "tts-1", description="TTS model to use. Defaults to 'tts-1'."
    )
    input: str = Field(
        ...,
        description="Text to generate audio for. If the input exceeds 4096 characters, it will be split into chunks.",
    )
    voice: Optional[
        Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    ] = Field("alloy", description="Voice to use.")
    response_format: Optional[
        Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
    ] = Field("mp3", description="Audio format.")
    speed: Optional[float] = Field(
        1.0, ge=0.25, le=4.0, description="Speed of the audio."
    )


class AudioTranscriptionRequest(BaseModel):
    model: Optional[Literal["whisper-1"]] = Field(
        "whisper-1", description="Model to use. Defaults to 'whisper-1'."
    )
    file: Union[
        bytes,
        BytesIO,
        IO[bytes],
        BufferedReader,
        str,
        Path,
        Tuple[Optional[str], bytes],
        Tuple[Optional[str], bytes, Optional[str]],
    ] = Field(..., description="Audio file content.")
    language: Optional[str] = Field(
        None, description="Language of the audio in ISO-639-1 format."
    )
    prompt: Optional[str] = Field(
        None, description="Optional prompt for the transcription."
    )
    response_format: Optional[
        Literal["json", "text", "srt", "verbose_json", "vtt"]
    ] = Field("json", description="Response format.")
    temperature: Optional[float] = Field(
        0.0, ge=0.0, le=1.0, description="Sampling temperature."
    )
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Field(
        None, description="Granularity of timestamps."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("file", mode="before")
    @classmethod
    def validate_file(
        cls, value: Union[bytes, IO, str, Path, Tuple]
    ) -> Union[IO[bytes], Tuple[str, bytes]]:
        """
        Ensure the file field is valid and prepare it for OpenAI API.
        """
        if isinstance(value, (str, Path)):
            # Convert file path to a readable file-like object
            try:
                return open(value, "rb")
            except Exception as e:
                raise ValueError(f"Invalid file path: {value}. Error: {e}")
        elif isinstance(value, bytes):
            # Wrap raw bytes with a default filename
            return "file.mp3", value
        elif isinstance(value, BufferedReader) or (
            hasattr(value, "read") and callable(value.read)
        ):
            # Allow BufferedReader or other file-like objects as-is
            return value
        elif isinstance(value, BufferedReader) or (
            hasattr(value, "read") and callable(value.read)
        ):
            if value.closed:
                raise ValueError("File-like object must remain open during request.")
            return value
        elif isinstance(value, tuple):
            # Handle tuples with (filename, bytes or file-like object)
            if len(value) == 2:
                filename, file_obj = value
                if isinstance(file_obj, bytes):
                    return filename, file_obj
                elif hasattr(file_obj, "read") and callable(file_obj.read):
                    return filename, file_obj.read()
            raise ValueError(
                "File tuple must be of the form (filename, bytes) or (filename, file-like object)."
            )
        else:
            raise ValueError(f"Unsupported file type: {type(value)}.")


class AudioTranslationRequest(BaseModel):
    model: Optional[Literal["whisper-1"]] = Field(
        "whisper-1", description="Model to use. Defaults to 'whisper-1'."
    )
    file: Union[
        bytes,
        BytesIO,
        IO[bytes],
        BufferedReader,
        str,
        Path,
        Tuple[Optional[str], bytes],
        Tuple[Optional[str], bytes, Optional[str]],
    ] = Field(..., description="Audio file content.")
    prompt: Optional[str] = Field(
        None, description="Optional prompt for the translation."
    )
    response_format: Optional[
        Literal["json", "text", "srt", "verbose_json", "vtt"]
    ] = Field("json", description="Response format.")
    temperature: Optional[float] = Field(
        0.0, ge=0.0, le=1.0, description="Sampling temperature."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("file", mode="before")
    @classmethod
    def validate_file(
        cls, value: Union[bytes, IO, str, Path, Tuple]
    ) -> Union[IO[bytes], Tuple[str, bytes]]:
        """
        Ensure the file field is valid and prepare it for OpenAI API.
        """
        if isinstance(value, (str, Path)):
            # Convert file path to a readable file-like object
            try:
                return open(value, "rb")
            except Exception as e:
                raise ValueError(f"Invalid file path: {value}. Error: {e}")
        elif isinstance(value, bytes):
            # Wrap raw bytes with a default filename
            return "file.mp3", value
        elif isinstance(value, BufferedReader) or (
            hasattr(value, "read") and callable(value.read)
        ):
            if value.closed:  # Reopen if closed
                raise ValueError("File-like object must remain open during request.")
            return value
        elif isinstance(value, tuple):
            # Handle tuples with (filename, bytes or file-like object)
            if len(value) == 2:
                filename, file_obj = value
                if isinstance(file_obj, bytes):
                    return filename, file_obj
                elif hasattr(file_obj, "read") and callable(file_obj.read):
                    return filename, file_obj.read()
            raise ValueError(
                "File tuple must be of the form (filename, bytes) or (filename, file-like object)."
            )
        else:
            raise ValueError(f"Unsupported file type: {type(value)}.")


class AudioTranscriptionResponse(BaseModel):
    text: str
    language: Optional[str]
    duration: Optional[float]
    segments: Optional[List[Dict[str, Union[str, float, List[int]]]]]


class AudioTranslationResponse(BaseModel):
    text: str
