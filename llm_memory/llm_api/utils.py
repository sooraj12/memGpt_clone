import requests

from llm_api.chat_completion_wrappers import llama3


def get_available_wrappers() -> dict:
    return {
        "llama3": llama3.LLaMA3InnerMonologueWrapper(),
        "llama3-grammar": llama3.LLaMA3InnerMonologueWrapper(),
        "llama3-hints-grammar": llama3.LLaMA3InnerMonologueWrapper(
            assistant_prefix_hint=True
        ),
    }


def post_json_auth_request(uri, json_payload, auth_type, auth_key):
    """Send a POST request with a JSON payload and optional authentication"""

    # By default most local LLM inference servers do not have authorization enabled
    if auth_type is None:
        response = requests.post(uri, json=json_payload)

    # Used by OpenAI, together.ai, Mistral AI
    elif auth_type == "bearer_token":
        if auth_key is None:
            raise ValueError(f"auth_type is {auth_type}, but auth_key is null")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_key}",
        }
        response = requests.post(uri, json=json_payload, headers=headers)

    # Used by OpenAI Azure
    elif auth_type == "api_key":
        if auth_key is None:
            raise ValueError(f"auth_type is {auth_type}, but auth_key is null")
        headers = {"Content-Type": "application/json", "api-key": f"{auth_key}"}
        response = requests.post(uri, json=json_payload, headers=headers)

    else:
        raise ValueError(f"Unsupported authentication type: {auth_type}")

    return response
