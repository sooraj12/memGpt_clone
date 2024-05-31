import os

JSON_ENSURE_ASCII = False

MEMGPT_DIR = os.path.join(".", ".memgpt")

DEFAULT_PERSONA = "sam_pov"
DEFAULT_HUMAN = "basic"
DEFAULT_PRESET = "memgpt_chat"

LLM_MAX_TOKENS = {
    "DEFAULT": 8192,
    "llama3-8b-8192": 8192,
}

CORE_MEMORY_PERSONA_CHAR_LIMIT = 2000
CORE_MEMORY_HUMAN_CHAR_LIMIT = 2000

CLI_WARNING_PREFIX = "Warning: "

FUNCTION_PARAM_NAME_REQ_HEARTBEAT = "request_heartbeat"
FUNCTION_PARAM_TYPE_REQ_HEARTBEAT = "boolean"
FUNCTION_PARAM_DESCRIPTION_REQ_HEARTBEAT = "Request an immediate heartbeat after function execution. Set to 'true' if you want to send a follow-up message or run a follow-up function."


MAX_PAUSE_HEARTBEATS = 360  # in min
RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE = 5
JSON_LOADS_STRICT = False

MESSAGE_CHATGPT_FUNCTION_MODEL = "llama3-8b-8192"
MESSAGE_CHATGPT_FUNCTION_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Keep your responses short and concise."
)

MAX_EMBEDDING_DIM = 4096  # maximum supported embedding size - do NOT change or else DBs will need to be reset
TOOL_CALL_ID_MAX_LEN = 29
INNER_THOUGHTS_KWARG = "inner_thoughts"

MAX_EMBEDDING_DIM = 4096

STARTUP_QUOTES = [
    "I think, therefore I am.",
    "All those moments will be lost in time, like tears in rain.",
    "More human than human is our motto.",
]

INITIAL_BOOT_MESSAGE_SEND_MESSAGE_FIRST_MSG = STARTUP_QUOTES[2]
INITIAL_BOOT_MESSAGE = "Boot sequence complete. Persona activated."
INITIAL_BOOT_MESSAGE_SEND_MESSAGE_THOUGHT = (
    "Bootup sequence complete. Persona activated. Testing messaging functionality."
)
FIRST_MESSAGE_ATTEMPTS = 10
INNER_THOUGHTS_KWARG_DESCRIPTION = "Deep inner monologue private to you only."

SETTINGS_FOLDER_NAME = "settings"
COMPLETION_SETTINGS_FILE_NAME = "completions_api_settings.json"

TOOL_CALL_ID_MAX_LEN = 29
FUNCTION_RETURN_CHAR_LIMIT = 3000
MESSAGE_SUMMARY_WARNING_FRAC = 0.75

MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST = 3
MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC = 0.75

MESSAGE_SUMMARY_REQUEST_ACK = "Understood, I will respond with a summary of the message (and only the summary, nothing else) once I receive the conversation history. I'm ready."

NON_USER_MSG_PREFIX = "[This is an automated system message hidden from the user] "
MESSAGE_SUMMARY_WARNING_STR = " ".join(
    [
        f"{NON_USER_MSG_PREFIX}The conversation history will soon reach its maximum length and be trimmed.",
        "Do NOT tell the user about this system alert, they should not know that the history is reaching max length.",
        "If there is any important new information or general memories about you or the user that you would like to save, you should save that information immediately by calling function core_memory_append, core_memory_replace, or archival_memory_insert.",
        # "Remember to pass request_heartbeat = true if you would like to send a message immediately after.",
    ]
)

REQ_HEARTBEAT_MESSAGE = f"{NON_USER_MSG_PREFIX}Function called using request_heartbeat=true, returning control"
FUNC_FAILED_HEARTBEAT_MESSAGE = (
    f"{NON_USER_MSG_PREFIX}Function call failed, returning control"
)
