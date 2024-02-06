from pathlib import Path

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.audio import Transcription
from .sdk.forge_log import ForgeLogger
from litellm import acompletion, AuthenticationError, InvalidRequestError, ModelResponse


LOG = ForgeLogger(__name__)


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(model, messages, **kwargs) -> ModelResponse:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        kwargs["model"] = model
        kwargs["messages"] = messages

        resp = await acompletion(**kwargs)

        return resp
    except AuthenticationError as e:
        LOG.exception("Authentication Error")
    except InvalidRequestError as e:
        LOG.exception("Invalid Request Error")
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def create_embedding_request(
    messages, model="text-embedding-ada-002"
) -> CreateEmbeddingResponse:
    """Generate an embedding for a list of messages using OpenAI's API"""
    try:
        return OpenAI().embeddings.create(
            input=[f"{m['role']}: {m['content']}" for m in messages],
            model=model,
        )
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def transcribe_audio(
    audio_file: Path,
) -> Transcription:
    """Transcribe an audio file using OpenAI's API"""
    try:
        return OpenAI().Audio.transcribe(model="whisper-1", file=audio_file)
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise
