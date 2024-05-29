import os
import inspect
import pytz
import json
import difflib
import uuid
import hashlib
import tiktoken

from datetime import timezone, datetime, timedelta
from functools import wraps
from typing import _GenericAlias, get_type_hints, Union
from urllib.parse import urlparse
import demjson3 as demjson
from constants import (
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
    MEMGPT_DIR,
    JSON_ENSURE_ASCII,
    TOOL_CALL_ID_MAX_LEN,
    JSON_LOADS_STRICT,
    CLI_WARNING_PREFIX,
    FUNCTION_RETURN_CHAR_LIMIT,
)


def get_utc_time() -> datetime:
    """Get the current UTC time"""
    return datetime.now(timezone.utc)


def is_utc_datetime(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) == timedelta(0)


def get_local_time_timezone(timezone="America/Los_Angeles"):
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone(timezone)
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire, including AM/PM
    formatted_time = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return formatted_time


def get_local_time(timezone=None):
    if timezone is not None:
        time_str = get_local_time_timezone(timezone)
    else:
        # Get the current time, which will be in the local timezone of the computer
        local_time = datetime.now().astimezone()

        # You may format it as you desire, including AM/PM
        time_str = local_time.strftime("%Y-%m-%d %I:%M:%S %p %Z%z")

    return time_str.strip()


def list_human_files():
    """List all humans files"""
    defaults_dir = os.path.join(".", "humans", "examples")
    user_dir = os.path.join(MEMGPT_DIR, "humans")

    memgpt_defaults = os.listdir(defaults_dir)
    memgpt_defaults = [
        os.path.join(defaults_dir, f) for f in memgpt_defaults if f.endswith(".txt")
    ]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return memgpt_defaults + user_added


def list_persona_files():
    """List all personas files"""
    defaults_dir = os.path.join(".", "personas", "examples")
    user_dir = os.path.join(MEMGPT_DIR, "personas")

    memgpt_defaults = os.listdir(defaults_dir)
    memgpt_defaults = [
        os.path.join(defaults_dir, f) for f in memgpt_defaults if f.endswith(".txt")
    ]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return memgpt_defaults + user_added


def get_human_text(name: str, enforce_limit=True):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            human_text = open(file_path, "r").read().strip()
            if enforce_limit and len(human_text) > CORE_MEMORY_HUMAN_CHAR_LIMIT:
                raise ValueError(
                    f"Contents of {name}.txt is over the character limit ({len(human_text)} > {CORE_MEMORY_HUMAN_CHAR_LIMIT})"
                )
            return human_text

    raise ValueError(f"Human {name}.txt not found")


def get_persona_text(name: str, enforce_limit=True):
    for file_path in list_persona_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            persona_text = open(file_path, "r").read().strip()
            if enforce_limit and len(persona_text) > CORE_MEMORY_PERSONA_CHAR_LIMIT:
                raise ValueError(
                    f"Contents of {name}.txt is over the character limit ({len(persona_text)} > {CORE_MEMORY_PERSONA_CHAR_LIMIT})"
                )
            return persona_text

    raise ValueError(f"Persona {name}.txt not found")


def is_optional_type(hint):
    """Check if the type hint is an Optional type."""
    if isinstance(hint, _GenericAlias):
        return hint.__origin__ is Union and type(None) in hint.__args__
    return False


def enforce_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints, excluding the return type hint
        hints = {k: v for k, v in get_type_hints(func).items() if k != "return"}

        # Get the function's argument names
        arg_names = inspect.getfullargspec(func).args

        # Pair each argument with its corresponding type hint
        args_with_hints = dict(zip(arg_names[1:], args[1:]))  # Skipping 'self'

        # Check types of arguments
        for arg_name, arg_value in args_with_hints.items():
            hint = hints.get(arg_name)
            if (
                hint
                and not isinstance(arg_value, hint)
                and not (is_optional_type(hint) and arg_value is None)
            ):
                raise ValueError(f"Argument {arg_name} does not match type {hint}")

        # Check types of keyword arguments
        for arg_name, arg_value in kwargs.items():
            hint = hints.get(arg_name)
            if (
                hint
                and not isinstance(arg_value, hint)
                and not (is_optional_type(hint) and arg_value is None)
            ):
                raise ValueError(f"Argument {arg_name} does not match type {hint}")

        return func(*args, **kwargs)

    return wrapper


def get_schema_diff(schema_a, schema_b):
    # Assuming f_schema and linked_function['json_schema'] are your JSON schemas
    f_schema_json = json.dumps(schema_a, indent=2, ensure_ascii=JSON_ENSURE_ASCII)
    linked_function_json = json.dumps(
        schema_b, indent=2, ensure_ascii=JSON_ENSURE_ASCII
    )

    # Compute the difference using difflib
    difference = list(
        difflib.ndiff(
            f_schema_json.splitlines(keepends=True),
            linked_function_json.splitlines(keepends=True),
        )
    )

    # Filter out lines that don't represent changes
    difference = [
        line for line in difference if line.startswith("+ ") or line.startswith("- ")
    ]

    return "".join(difference)


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def create_uuid_from_string(val: str):
    """
    Generate consistent UUID from a string
    from: https://samos-it.com/posts/python-create-uuid-from-random-string-of-words.html
    """
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def datetime_to_timestamp(dt):
    # convert datetime object to integer timestamp
    return int(dt.timestamp())


def timestamp_to_datetime(ts):
    # convert integer timestamp to datetime object
    return datetime.fromtimestamp(ts)


def count_tokens(s: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def get_tool_call_id() -> str:
    return str(uuid.uuid4())[:TOOL_CALL_ID_MAX_LEN]


def parse_json(string) -> dict:
    """Parse JSON string into JSON with both json and demjson"""
    result = None
    try:
        result = json.loads(string, strict=JSON_LOADS_STRICT)
        return result
    except Exception as e:
        print(f"Error parsing json with json package: {e}")

    try:
        result = demjson.decode(string)
        return result
    except demjson.JSONDecodeError as e:
        print(f"Error parsing json with demjson package: {e}")
        raise e


def validate_function_response(
    function_response_string: any, strict: bool = False, truncate: bool = True
) -> str:
    """Check to make sure that a function used by MemGPT returned a valid response

    Responses need to be strings (or None) that fall under a certain text count limit.
    """
    if not isinstance(function_response_string, str):
        # Soft correction for a few basic types

        if function_response_string is None:
            # function_response_string = "Empty (no function output)"
            function_response_string = "None"  # backcompat

        elif isinstance(function_response_string, dict):
            if strict:
                # TODO add better error message
                raise ValueError(function_response_string)

            # Allow dict through since it will be cast to json.dumps()
            try:
                # TODO find a better way to do this that won't result in double escapes
                function_response_string = json.dumps(
                    function_response_string, ensure_ascii=JSON_ENSURE_ASCII
                )
            except Exception:
                raise ValueError(function_response_string)

        else:
            if strict:
                # TODO add better error message
                raise ValueError(function_response_string)

            # Try to convert to a string, but throw a warning to alert the user
            try:
                function_response_string = str(function_response_string)
            except Exception:
                raise ValueError(function_response_string)

    # Now check the length and make sure it doesn't go over the limit
    # TODO we should change this to a max token limit that's variable based on tokens remaining (or context-window)
    if truncate and len(function_response_string) > FUNCTION_RETURN_CHAR_LIMIT:
        print(
            f"{CLI_WARNING_PREFIX}function return was over limit ({len(function_response_string)} > {FUNCTION_RETURN_CHAR_LIMIT}) and was truncated"
        )
        function_response_string = f"{function_response_string[:FUNCTION_RETURN_CHAR_LIMIT]}... [NOTE: function output was truncated since it exceeded the character limit ({len(function_response_string)} > {FUNCTION_RETURN_CHAR_LIMIT})]"

    return function_response_string
