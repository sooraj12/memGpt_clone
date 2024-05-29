import requests
import json
import uuid

from models.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    Message,
    ToolCall,
    UsageStatistics,
)
from llm_api.utils import get_available_wrappers
from llm_api.chat_completion_wrappers import simple_summary_wrapper
from prompts.gpt_summarize import SYSTEM as SUMMARIZE_SYSTEM_MESSAGE
from errors import LocalLLMError, LocalLLMConnectionError
from llm_api.groq.api import get_groq_completion
from llm_api.function_parser import patch_function
from utils import count_tokens, get_utc_time, get_tool_call_id
from constants import JSON_ENSURE_ASCII


has_shown_warning = False


def get_chat_completion(
    model,
    # no model required (except for Ollama), since the model is fixed to whatever you set in your own backend
    messages,
    functions=None,
    function_call="auto",
    context_window=None,
    user=None,
    # required
    wrapper=None,
    endpoint=None,
    # optional cleanup
    function_correction=True,
    # extra hints to allow for additional prompt formatting hacks
    first_message=False,
) -> ChatCompletionResponse:
    assert (
        context_window is not None
    ), "Local LLM calls need the context length to be explicitly set"
    assert (
        endpoint is not None
    ), "Local LLM calls need the endpoint (eg http://localendpoint:1234) to be explicitly set"
    global has_shown_warning

    if not isinstance(messages[0], dict):
        messages = [m.to_openai_dict() for m in messages]

    if function_call is not None and function_call != "auto":
        raise ValueError(
            f"function_call == {function_call} not supported (auto or None only)"
        )

    available_wrappers = get_available_wrappers()

    # Special case for if the call we're making is coming from the summarizer
    if (
        messages[0]["role"] == "system"
        and messages[0]["content"].strip() == SUMMARIZE_SYSTEM_MESSAGE.strip()
    ):
        llm_wrapper = simple_summary_wrapper.SimpleSummaryWrapper()

    else:
        llm_wrapper = available_wrappers[wrapper]

    # First step: turn the message sequence into a prompt that the model expects
    try:
        if (
            hasattr(llm_wrapper, "supports_first_message")
            and llm_wrapper.supports_first_message
        ):
            prompt = llm_wrapper.chat_completion_to_prompt(
                messages=messages,
                functions=functions,
                first_message=first_message,
            )
        else:
            prompt = llm_wrapper.chat_completion_to_prompt(
                messages=messages,
                functions=functions,
            )
    except Exception as e:
        print(e)
        raise LocalLLMError(
            f"Failed to convert ChatCompletion messages into prompt string with wrapper {str(llm_wrapper)} - error: {str(e)}"
        )

    try:
        result, usage = get_groq_completion(
            endpoint,
            model,
            prompt,
            context_window,
            "bearer_token",
            "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",
        )
    except requests.exceptions.ConnectionError:
        raise LocalLLMConnectionError(f"Unable to connect to endpoint {endpoint}")

    if result is None or result == "":
        raise LocalLLMError(f"Got back an empty response string from {endpoint}")

    try:
        if (
            hasattr(llm_wrapper, "supports_first_message")
            and llm_wrapper.supports_first_message
        ):
            chat_completion_result = llm_wrapper.output_to_chat_completion_response(
                result, first_message=first_message
            )
        else:
            chat_completion_result = llm_wrapper.output_to_chat_completion_response(
                result
            )
    except Exception as e:
        raise LocalLLMError(
            f"Failed to parse JSON from local LLM response - error: {str(e)}"
        )

    # Run through some manual function correction (optional)
    if function_correction:
        chat_completion_result = patch_function(
            message_history=messages, new_message=chat_completion_result
        )

    # Fill in potential missing usage information (used for tracking token use)
    if not (
        "prompt_tokens" in usage
        and "completion_tokens" in usage
        and "total_tokens" in usage
    ):
        raise LocalLLMError(f"usage dict in response was missing fields ({usage})")

    if usage["prompt_tokens"] is None:
        print("usage dict was missing prompt_tokens, computing on-the-fly...")
        usage["prompt_tokens"] = count_tokens(prompt)

    # NOTE: we should compute on-the-fly anyways since we might have to correct for errors during JSON parsing
    usage["completion_tokens"] = count_tokens(
        json.dumps(chat_completion_result, ensure_ascii=JSON_ENSURE_ASCII)
    )

    # NOTE: this is the token count that matters most
    if usage["total_tokens"] is None:
        print("usage dict was missing total_tokens, computing on-the-fly...")
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    # unpack with response.choices[0].message.content
    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=Message(
                    role=chat_completion_result["role"],
                    content=chat_completion_result["content"],
                    tool_calls=(
                        [
                            ToolCall(
                                id=get_tool_call_id(),
                                type="function",
                                function=chat_completion_result["function_call"],
                            )
                        ]
                        if "function_call" in chat_completion_result
                        else []
                    ),
                ),
            )
        ],
        created=get_utc_time(),
        model=model,
        # "This fingerprint represents the backend configuration that the model runs with."
        # system_fingerprint=user if user is not None else "null",
        system_fingerprint=None,
        object="chat.completion",
        usage=UsageStatistics(**usage),
    )

    return response
