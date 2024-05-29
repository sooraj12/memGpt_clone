import requests
import random
import time
import uuid

from constants import CLI_WARNING_PREFIX
from typing import List
from models.chat_completion_response import ChatCompletionResponse
from data_types import Message
from models.pydantic_models import LLMConfigModel
from llm_api.chat_completion_proxy import get_chat_completion


def is_context_overflow_error(exception: requests.exceptions.RequestException) -> bool:
    """Checks if an exception is due to context overflow (based on common OpenAI response messages)"""

    match_string = "maximum context length"

    if match_string in str(exception):
        print(f"Found '{match_string}' in str(exception)={(str(exception))}")
        return True

    elif isinstance(exception, requests.exceptions.HTTPError):
        if (
            exception.response is not None
            and "application/json" in exception.response.headers.get("Content-Type", "")
        ):
            try:
                error_details = exception.response.json()
                if "error" not in error_details:
                    print(
                        f"HTTPError occurred, but couldn't find error field: {error_details}"
                    )
                    return False
                else:
                    error_details = error_details["error"]

                # Check for the specific error code
                if error_details.get("code") == "context_length_exceeded":
                    print(
                        f"HTTPError occurred, caught error code {error_details.get('code')}"
                    )
                    return True
                # Soft-check for "maximum context length" inside of the message
                elif error_details.get(
                    "message"
                ) and "maximum context length" in error_details.get("message"):
                    print(
                        f"HTTPError occurred, found '{match_string}' in error message contents ({error_details})"
                    )
                    return True
                else:
                    print(
                        f"HTTPError occurred, but unknown error message: {error_details}"
                    )
                    return False
            except ValueError:
                # JSON decoding failed
                print(f"HTTPError occurred ({exception}), but no JSON error message.")

    # Generic fail
    else:
        return False


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    # List of OpenAI error codes: https://github.com/openai/openai-python/blob/17ac6779958b2b74999c634c4ea4c7b74906027a/src/openai/_client.py#L227-L250
    # 429 = rate limit
    error_codes: tuple = (429,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        pass

        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            except requests.exceptions.HTTPError as http_err:
                # Retry on specified errors
                if http_err.response.status_code in error_codes:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    # print(f"Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying...")
                    print(
                        f"{CLI_WARNING_PREFIX}Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying..."
                    )
                    time.sleep(delay)
                else:
                    # For other HTTP errors, re-raise the exception
                    raise

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def create(
    llm_config: LLMConfigModel,
    messages: List[Message],
    user_id: uuid.UUID = None,  # option UUID to associate request with
    functions: list = None,
    function_call: str = "auto",
    # hint
    first_message: bool = False,
) -> ChatCompletionResponse:
    """Return response to chat completion with backoff"""

    print(f"Using model endpoint: {llm_config.model_endpoint}")

    if function_call and not functions:
        print("un-setting function_call because functions is None")
        function_call = None

    return get_chat_completion(
        model=llm_config.model,
        messages=messages,
        functions=functions,
        function_call=function_call,
        context_window=llm_config.context_window,
        endpoint=llm_config.model_endpoint,
        wrapper=llm_config.model_wrapper,
        user=str(user_id),
        # hint
        first_message=first_message,
    )
