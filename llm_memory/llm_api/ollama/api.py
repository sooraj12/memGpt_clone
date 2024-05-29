from urllib.parse import urljoin
from errors import LocalLLMError
from utils import count_tokens
from llm_api.settings.settings import get_completions_settings
from llm_api.utils import post_json_auth_request

OLLAMA_API_SUFFIX = "/api/generate"


def get_ollama_completion(endpoint, model, prompt, context_window):
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(
            f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)"
        )

    if model is None:
        raise LocalLLMError(
            "Error: model name not specified. Set model in your config to the model you want to run (e.g. 'llama3-70b')"
        )

    # Settings for the generation, includes the prompt + stop tokens, max length, etc
    # https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
    settings = get_completions_settings()
    settings.update(
        {
            # specific naming for context length
            "num_ctx": context_window,
        }
    )

    # https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-completion
    request = {
        ## base parameters
        "model": model,
        "prompt": prompt,
        ## advanced parameters
        "stream": False,
        "options": settings,
        # "raw mode does not support template, system, or context"
        "raw": True,  # no prompt formatting
    }

    try:
        URI = urljoin(endpoint.strip("/") + "/", OLLAMA_API_SUFFIX.strip("/"))
        response = post_json_auth_request(uri=URI, json_payload=request)
        if response.status_code == 200:
            # https://github.com/jmorganca/ollama/blob/main/docs/api.md
            result_full = response.json()
            print(f"JSON API response:\n{result_full}")
            result = result_full["response"]
        else:
            raise Exception(
                f"API call got non-200 response code (code={response.status_code}, msg={response.text}) for address: {URI}."
                + f" Make sure that the ollama API server is running and reachable at {URI}."
            )

    except:
        raise

    # Pass usage statistics back to main thread
    # These are used to compute memory warning messages
    # https://github.com/jmorganca/ollama/blob/main/docs/api.md#response
    completion_tokens = result_full.get("eval_count", None)
    total_tokens = (
        prompt_tokens + completion_tokens if completion_tokens is not None else None
    )
    usage = {
        "prompt_tokens": prompt_tokens,  # can also grab from "prompt_eval_count"
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return result, usage
