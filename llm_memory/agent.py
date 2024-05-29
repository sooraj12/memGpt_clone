import uuid
import datetime
import json
import inspect
import traceback


from metadata import MetadataStore
from interface import QueuingInterface
from typing import Optional, List, cast, Union, Tuple
from data_types import Preset, LLMConfig, EmbeddingConfig, Message, AgentState
from utils import (
    is_utc_datetime,
    get_schema_diff,
    get_local_time,
    get_utc_time,
    get_tool_call_id,
    parse_json,
    validate_function_response,
    count_tokens,
)
from functions.functions import load_all_function_sets
from memory import CoreMemory, ArchivalMemory, RecallMemory, summarize_messages
from persistance_manager import LocalStateManager
from system import (
    get_login_event,
    get_initial_boot_messages,
    package_function_response,
    package_summarize_message,
)
from llm_api.llm_api_tools import is_context_overflow_error
from models import chat_completion_response
from llm_api.llm_api_tools import create
from errors import LLMError
from constants import (
    JSON_ENSURE_ASCII,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
    FIRST_MESSAGE_ATTEMPTS,
    CLI_WARNING_PREFIX,
    JSON_LOADS_STRICT,
    LLM_MAX_TOKENS,
    MESSAGE_SUMMARY_WARNING_FRAC,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
)


def link_functions(function_schemas: list):
    """Link function definitions to list of function schemas"""

    # need to dynamically link the functions
    # the saved agent.functions will just have the schemas, but we need to
    # go through the functions library and pull the respective python functions

    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    # agent.functions is a list of schemas (OpenAI kwarg functions style, see: https://platform.openai.com/docs/api-reference/chat/create)
    # [{'name': ..., 'description': ...}, {...}]
    available_functions = load_all_function_sets()
    linked_function_set = {}
    for f_schema in function_schemas:
        # Attempt to find the function in the existing function library
        f_name = f_schema.get("name")
        if f_name is None:
            raise ValueError(
                f"While loading agent.state.functions encountered a bad function schema object with no name:\n{f_schema}"
            )
        linked_function = available_functions.get(f_name)
        if linked_function is None:
            raise ValueError(
                f"Function '{f_name}' was specified in agent.state.functions, but is not in function library:\n{available_functions.keys()}"
            )
        # Once we find a matching function, make sure the schema is identical
        if json.dumps(f_schema, ensure_ascii=JSON_ENSURE_ASCII) != json.dumps(
            linked_function["json_schema"], ensure_ascii=JSON_ENSURE_ASCII
        ):
            # error_message = (
            #     f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different."
            #     + f"\n>>>agent.state.functions\n{json.dumps(f_schema, indent=2, ensure_ascii=JSON_ENSURE_ASCII)}"
            #     + f"\n>>>function library\n{json.dumps(linked_function['json_schema'], indent=2, ensure_ascii=JSON_ENSURE_ASCII)}"
            # )
            schema_diff = get_schema_diff(f_schema, linked_function["json_schema"])
            error_message = (
                f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different.\n"
                + "".join(schema_diff)
            )

            # NOTE to handle old configs, instead of erroring here let's just warn
            # raise ValueError(error_message)
            print(error_message)
        linked_function_set[f_name] = linked_function
    return linked_function_set


def initialize_memory(ai_notes: Union[str, None], human_notes: Union[str, None]):
    memory = CoreMemory(
        human_char_limit=CORE_MEMORY_HUMAN_CHAR_LIMIT,
        persona_char_limit=CORE_MEMORY_PERSONA_CHAR_LIMIT,
    )

    memory.edit_persona(ai_notes)
    memory.edit_human(human_notes)
    return memory


def construct_system_with_memory(
    system: str,
    memory: CoreMemory,
    memory_edit_timestamp: str,
    archival_memory: Optional[ArchivalMemory] = None,
    recall_memory: Optional[RecallMemory] = None,
    include_char_count: bool = True,
):
    full_system_message = "\n".join(
        [
            system,
            "\n",
            f"### Memory [last modified: {memory_edit_timestamp.strip()}]",
            f"{len(recall_memory) if recall_memory else 0} previous messages between you and the user are stored in recall memory (use functions to access them)",
            f"{len(archival_memory) if archival_memory else 0} total memories you created are stored in archival memory (use functions to access them)",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
            f'<persona characters="{len(memory.persona)}/{memory.persona_char_limit}">'
            if include_char_count
            else "<persona>",
            memory.persona,
            "</persona>",
            f'<human characters="{len(memory.human)}/{memory.human_char_limit}">'
            if include_char_count
            else "<human>",
            memory.human,
            "</human>",
        ]
    )
    return full_system_message


def initialize_message_sequence(
    model: str,
    system: str,
    memory: CoreMemory,
    archival_memory: Optional[ArchivalMemory] = None,
    recall_memory: Optional[RecallMemory] = None,
    memory_edit_timestamp: Optional[str] = None,
    include_initial_boot_message: bool = True,
) -> List[dict]:
    if memory_edit_timestamp is None:
        memory_edit_timestamp = get_local_time()

    full_system_message = construct_system_with_memory(
        system,
        memory,
        memory_edit_timestamp,
        archival_memory=archival_memory,
        recall_memory=recall_memory,
    )
    first_user_message = (
        get_login_event()
    )  # event letting MemGPT know the user just logged in

    if include_initial_boot_message:
        initial_boot_messages = get_initial_boot_messages("startup_with_send_message")

        messages = (
            [
                {"role": "system", "content": full_system_message},
            ]
            + initial_boot_messages
            + [
                {"role": "user", "content": first_user_message},
            ]
        )

    else:
        messages = [
            {"role": "system", "content": full_system_message},
            {"role": "user", "content": first_user_message},
        ]

    return messages


class Agent:
    def __init__(
        self,
        interface: QueuingInterface,
        # agents can be created from providing agent_state
        agent_state: Optional[AgentState] = None,
        # or from providing a preset (requires preset + extra fields)
        preset: Optional[Preset] = None,
        created_by: Optional[uuid.UUID] = None,
        name: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        # extras
        messages_total: Optional[int] = None,
        first_message_verify_mono: bool = True,
    ):
        init_agent_state = None
        if preset is not None:
            init_agent_state = AgentState(
                name="Sooraj",
                user_id=created_by,
                persona=preset.persona,
                human=preset.human,
                llm_config=llm_config,
                embedding_config=embedding_config,
                preset=preset.name,
                state={
                    "persona": preset.persona,
                    "human": preset.human,
                    "system": preset.system,
                    "functions": preset.functions_schema,
                    "messages": None,
                },
            )

        elif agent_state is not None:
            init_agent_state = agent_state

        else:
            raise ValueError(
                "Both Preset and AgentState were null (must provide one or the other)"
            )

        self.agent_state = init_agent_state
        self.model = self.agent_state.llm_config.model
        self.system = self.agent_state.state["system"]
        self.functions = self.agent_state.state["functions"]
        self.functions_python = {
            k: v["python_function"]
            for k, v in link_functions(function_schemas=self.functions).items()
        }
        self.memory = initialize_memory(
            ai_notes=self.agent_state.state["persona"],
            human_notes=self.agent_state.state["human"],
        )
        self.interface = interface

        self.persistence_manager = LocalStateManager(agent_state=self.agent_state)

        self.pause_heartbeats_start = None
        self.pause_heartbeats_minutes = 0

        self.first_message_verify_mono = first_message_verify_mono

        self.agent_alerted_about_memory_pressure = False

        self._messages: List[Message] = []

        if (
            "messages" in self.agent_state.state
            and self.agent_state.state["messages"] is not None
        ):
            # Convert to IDs, and pull from the database
            raw_messages = [
                self.persistence_manager.recall_memory.storage.get(id=uuid.UUID(msg_id))
                for msg_id in self.agent_state.state["messages"]
            ]
            self._messages.extend(
                [cast(Message, msg) for msg in raw_messages if msg is not None]
            )

            for m in self._messages:
                if not is_utc_datetime(m.created_at):
                    print(
                        f"Warning - created_at on message for agent {self.agent_state.name} isn't UTC (text='{m.text}')"
                    )
                    m.created_at = m.created_at.replace(tzinfo=datetime.timezone.utc)

        else:
            init_messages = initialize_message_sequence(
                self.model,
                self.system,
                self.memory,
            )
            init_messages_objs = []
            for msg in init_messages:
                init_messages_objs.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict=msg,
                    )
                )
            self.messages_total = 0
            self._append_to_messages(
                added_messages=[
                    cast(Message, msg) for msg in init_messages_objs if msg is not None
                ]
            )

            for m in self._messages:
                if not is_utc_datetime(m.created_at):
                    print(
                        f"Warning - created_at on message for agent {self.agent_state.name} isn't UTC (text='{m.text}')"
                    )
                    m.created_at = m.created_at.replace(tzinfo=datetime.timezone.utc)

        self.messages_total = (
            messages_total if messages_total is not None else (len(self._messages) - 1)
        )
        self.messages_total_init = len(self._messages) - 1

        print(f"Agent initialized, self.messages_total={self.messages_total}")

        self.update_state()

    def _append_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.append to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.append_to_messages(added_messages)

        new_messages = self._messages + added_messages  # append

        self._messages = new_messages
        self.messages_total += len(added_messages)

    def update_state(self) -> AgentState:
        updated_state = {
            "persona": self.memory.persona,
            "human": self.memory.human,
            "system": self.system,
            "functions": self.functions,
            "messages": [str(msg.id) for msg in self._messages],
        }

        self.agent_state = AgentState(
            name=self.agent_state.name,
            user_id=self.agent_state.user_id,
            persona=self.agent_state.persona,
            human=self.agent_state.human,
            llm_config=self.agent_state.llm_config,
            embedding_config=self.agent_state.embedding_config,
            preset=self.agent_state.preset,
            id=self.agent_state.id,
            created_at=self.agent_state.created_at,
            state=updated_state,
        )
        return self.agent_state

    def summarize_messages_inplace(
        self, cutoff=None, preserve_last_N_messages=True, disallow_tool_as_first=True
    ):
        assert (
            self.messages[0]["role"] == "system"
        ), f"self.messages[0] should be system (instead got {self.messages[0]})"

        # Start at index 1 (past the system message),
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # Do not allow truncation of the last N messages, since these are needed for in-context examples of function calling
        token_counts = [count_tokens(str(msg)) for msg in self.messages]
        message_buffer_token_count = sum(token_counts[1:])  # no system message
        desired_token_count_to_summarize = int(
            message_buffer_token_count * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC
        )
        candidate_messages_to_summarize = self.messages[1:]
        token_counts = token_counts[1:]

        if preserve_last_N_messages:
            candidate_messages_to_summarize = candidate_messages_to_summarize[
                :-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST
            ]
            token_counts = token_counts[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]

        print(f"MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC={MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC}")
        print(f"MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}")
        print(f"token_counts={token_counts}")
        print(f"message_buffer_token_count={message_buffer_token_count}")
        print(f"desired_token_count_to_summarize={desired_token_count_to_summarize}")
        print(
            f"len(candidate_messages_to_summarize)={len(candidate_messages_to_summarize)}"
        )

        # If at this point there's nothing to summarize, throw an error
        if len(candidate_messages_to_summarize) == 0:
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(self.messages)}, preserve_N={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}]"
            )

        # Walk down the message buffer (front-to-back) until we hit the target token count
        tokens_so_far = 0
        cutoff = 0
        for i, msg in enumerate(candidate_messages_to_summarize):
            cutoff = i
            tokens_so_far += token_counts[i]
            if tokens_so_far > desired_token_count_to_summarize:
                break
        # Account for system message
        cutoff += 1

        # Try to make an assistant message come after the cutoff
        try:
            print(f"Selected cutoff {cutoff} was a 'user', shifting one...")
            if self.messages[cutoff]["role"] == "user":
                new_cutoff = cutoff + 1
                if self.messages[new_cutoff]["role"] == "user":
                    print(f"Shifted cutoff {new_cutoff} is still a 'user', ignoring...")
                cutoff = new_cutoff
        except IndexError:
            pass

        # Make sure the cutoff isn't on a 'tool' or 'function'
        if disallow_tool_as_first:
            while self.messages[cutoff]["role"] in [
                "tool",
                "function",
            ] and cutoff < len(self.messages):
                print(f"Selected cutoff {cutoff} was a 'tool', shifting one...")
                cutoff += 1

        message_sequence_to_summarize = self._messages[
            1:cutoff
        ]  # do NOT get rid of the system message
        if len(message_sequence_to_summarize) <= 1:
            # This prevents a potential infinite loop of summarizing the same message over and over
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(message_sequence_to_summarize)} <= 1]"
            )
        else:
            print(
                f"Attempting to summarize {len(message_sequence_to_summarize)} messages [1:{cutoff}] of {len(self._messages)}"
            )

        # We can't do summarize logic properly if context_window is undefined
        if self.agent_state.llm_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            print(
                f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}"
            )
            print(f"{self.agent_state}")
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS[self.model]
                if (self.model is not None and self.model in LLM_MAX_TOKENS)
                else LLM_MAX_TOKENS["DEFAULT"]
            )
        summary = summarize_messages(
            agent_state=self.agent_state,
            message_sequence_to_summarize=message_sequence_to_summarize,
        )
        print(f"Got summary: {summary}")

        # Metadata that's useful for the agent to see
        all_time_message_count = self.messages_total
        remaining_message_count = len(self.messages[cutoff:])
        hidden_message_count = all_time_message_count - remaining_message_count
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(
            summary, summary_message_count, hidden_message_count, all_time_message_count
        )
        print(f"Packaged into message: {summary_message}")

        prior_len = len(self.messages)
        self._trim_messages(cutoff)
        packed_summary_message = {"role": "user", "content": summary_message}
        self._prepend_to_messages(
            [
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=packed_summary_message,
                )
            ]
        )

        # reset alert
        self.agent_alerted_about_memory_pressure = False

        print(f"Ran summarizer, messages length {prior_len} -> {len(self.messages)}")

    def _handle_ai_response(
        self,
        response_message: chat_completion_response.Message,
        override_tool_call_id: bool = True,
    ) -> Tuple[List[Message], bool, bool]:
        """Handles parsing and function execution"""

        messages = []  # append these to the history when done

        # Step 2: check if LLM wanted to call a function
        if response_message.function_call or (
            response_message.tool_calls is not None
            and len(response_message.tool_calls) > 0
        ):
            if (
                response_message.tool_calls is not None
                and len(response_message.tool_calls) > 1
            ):
                print(
                    f">1 tool call not supported, using index=0 only\n{response_message.tool_calls}"
                )
                response_message.tool_calls = [response_message.tool_calls[0]]

            # generate UUID for tool call
            if override_tool_call_id or response_message.function_call:
                tool_call_id = get_tool_call_id()  # needs to be a string for JSON
                response_message.tool_calls[0].id = tool_call_id
            else:
                tool_call_id = response_message.tool_calls[0].id
                assert tool_call_id is not None  # should be defined

            # role: assistant (requesting tool call, set tool call ID)
            messages.append(
                # NOTE: we're recreating the message here
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            print(f"Function call message: {messages[-1]}")

            # The content if then internal monologue, not chat
            self.interface.internal_monologue(
                response_message.content, msg_obj=messages[-1]
            )

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors

            # Failure case 1: function name is wrong
            function_call = response_message.tool_calls[0].function
            function_name = function_call.name
            print(
                f"Request to call function {function_name} with tool_call_id: {tool_call_id}"
            )
            try:
                function_to_call = self.functions_python[function_name]
            except KeyError:
                error_msg = f"No function named {function_name}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(
                    f"Error: {error_msg}", msg_obj=messages[-1]
                )
                return (
                    messages,
                    False,
                    True,
                )  # force a heartbeat to allow agent to handle error

            # Failure case 2: function name is OK, but function args are bad JSON
            try:
                raw_function_args = function_call.arguments
                function_args = parse_json(raw_function_args)
            except Exception:
                error_msg = f"Error parsing JSON for function '{function_name}' arguments: {function_call.arguments}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(
                    f"Error: {error_msg}", msg_obj=messages[-1]
                )
                return (
                    messages,
                    False,
                    True,
                )  # force a heartbeat to allow agent to handle error

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            heartbeat_request = function_args.pop("request_heartbeat", None)
            if not (isinstance(heartbeat_request, bool) or heartbeat_request is None):
                print(
                    f"{CLI_WARNING_PREFIX}'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )
                heartbeat_request = False

            # Failure case 3: function failed during execution
            # NOTE: the msg_obj associated with the "Running " message is the prior assistant message, not the function/tool role message
            #       this is because the function/tool role message is only created once the function/tool has executed/returned
            self.interface.function_message(
                f"Running {function_name}({function_args})", msg_obj=messages[-1]
            )
            try:
                spec = inspect.getfullargspec(function_to_call).annotations
                for name, arg in function_args.items():
                    if isinstance(function_args[name], dict):
                        function_args[name] = spec[name](**function_args[name])

                function_args["self"] = (
                    self  # need to attach self to arg since it's dynamically linked
                )

                function_response = function_to_call(**function_args)
                if function_name in [
                    "conversation_search",
                    "conversation_search_date",
                    "archival_memory_search",
                ]:
                    # with certain functions we rely on the paging mechanism to handle overflow
                    truncate = False
                else:
                    # but by default, we add a truncation safeguard to prevent bad functions from
                    # overflow the agent context window
                    truncate = True
                function_response_string = validate_function_response(
                    function_response, truncate=truncate
                )
                function_args.pop("self", None)
                function_response = package_function_response(
                    True, function_response_string
                )
                function_failed = False
            except Exception as e:
                function_args.pop("self", None)
                error_msg = f"Error calling function {function_name}: {str(e)}"
                error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                print(error_msg_user)
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(
                    f"Ran {function_name}({function_args})", msg_obj=messages[-1]
                )
                self.interface.function_message(
                    f"Error: {error_msg}", msg_obj=messages[-1]
                )
                return (
                    messages,
                    False,
                    True,
                )  # force a heartbeat to allow agent to handle error

            # If no failures happened along the way: ...
            # Step 4: send the info on the function call and function response to GPT
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                        "tool_call_id": tool_call_id,
                    },
                )
            )  # extend conversation with function response
            self.interface.function_message(
                f"Ran {function_name}({function_args})", msg_obj=messages[-1]
            )
            self.interface.function_message(
                f"Success: {function_response_string}", msg_obj=messages[-1]
            )

        else:
            # Standard non-function reply
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            self.interface.internal_monologue(
                response_message.content, msg_obj=messages[-1]
            )
            heartbeat_request = False
            function_failed = False

        return messages, heartbeat_request, function_failed

    def _get_ai_reply(
        self,
        message_sequence: List[Message],
        function_call: str = "auto",
        first_message: bool = False,  # hint
    ) -> chat_completion_response.ChatCompletionResponse:
        """Get response from LLM API"""
        try:
            response = create(
                llm_config=self.agent_state.llm_config,
                user_id=self.agent_state.user_id,
                messages=message_sequence,
                functions=self.functions,
                function_call=function_call,
                # hint
                first_message=first_message,
            )
            # special case for 'length'
            if response.choices[0].finish_reason == "length":
                raise Exception("Finish reason was length (maximum context length)")

            # catches for soft errors
            if response.choices[0].finish_reason not in [
                "stop",
                "function_call",
                "tool_calls",
            ]:
                raise Exception(f"API call finish with bad finish reason: {response}")

            # unpack with response.choices[0].message.content
            return response
        except Exception as e:
            raise e

    def step(
        self,
        user_message: Union[Message, str],
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,  # if True, return dicts, if False, return Message objects
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
    ) -> Tuple[List[Union[dict, Message]], bool, bool, bool]:
        """Top-level event message handler for the agent"""

        def strip_name_field_from_user_message(
            user_message_text: str,
        ) -> Tuple[str, Optional[str]]:
            """If 'name' exists in the JSON string, remove it and return the cleaned text + name value"""
            try:
                user_message_json = dict(
                    json.loads(user_message_text, strict=JSON_LOADS_STRICT)
                )
                # Special handling for AutoGen messages with 'name' field
                # Treat 'name' as a special field
                # If it exists in the input message, elevate it to the 'message' level
                name = user_message_json.pop("name", None)
                clean_message = json.dumps(
                    user_message_json, ensure_ascii=JSON_ENSURE_ASCII
                )

            except Exception as e:
                print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed with: {e}")

            return clean_message, name

        def validate_json(user_message_text: str, raise_on_error: bool) -> str:
            try:
                user_message_json = dict(
                    json.loads(user_message_text, strict=JSON_LOADS_STRICT)
                )
                user_message_json_val = json.dumps(
                    user_message_json, ensure_ascii=JSON_ENSURE_ASCII
                )
                return user_message_json_val
            except Exception as e:
                print(
                    f"{CLI_WARNING_PREFIX}couldn't parse user input message as JSON: {e}"
                )
                if raise_on_error:
                    raise e

        try:
            # Step 0: add user message
            if user_message is not None:
                if isinstance(user_message, Message):
                    # Validate JSON via save/load
                    user_message_text = validate_json(user_message.text, False)
                    cleaned_user_message_text, name = (
                        strip_name_field_from_user_message(user_message_text)
                    )

                    if name is not None:
                        # Update Message object
                        user_message.text = cleaned_user_message_text
                        user_message.name = name

                    # Recreate timestamp
                    if recreate_message_timestamp:
                        user_message.created_at = get_utc_time()

                elif isinstance(user_message, str):
                    # Validate JSON via save/load
                    user_message = validate_json(user_message, False)
                    cleaned_user_message_text, name = (
                        strip_name_field_from_user_message(user_message)
                    )

                    # If user_message['name'] is not None, it will be handled properly by dict_to_message
                    # So no need to run strip_name_field_from_user_message
                    # Create the associated Message object (in the database)
                    user_message = Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "user",
                            "content": cleaned_user_message_text,
                            "name": name,
                        },
                    )

                else:
                    raise ValueError(f"Bad type for user_message: {type(user_message)}")

                self.interface.user_message(user_message.text, msg_obj=user_message)
                input_message_sequence = self._messages + [user_message]
            # Alternatively, the requestor can send an empty user message
            else:
                input_message_sequence = self._messages

            if (
                len(input_message_sequence) > 1
                and input_message_sequence[-1].role != "user"
            ):
                print(
                    f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue"
                )

            # Step 1: send the conversation and available functions to GPT
            if not skip_verify and (
                first_message or self.messages_total == self.messages_total_init
            ):
                print(
                    "This is the first message. Running extra verifier on AI response."
                )
                counter = 0

                while True:
                    response = self._get_ai_reply(
                        message_sequence=input_message_sequence,
                        first_message=True,  # passed through to the prompt formatter
                    )

                    counter += 1

                    if counter > first_message_retry_limit:
                        raise Exception(
                            f"Hit first message retry limit ({first_message_retry_limit})"
                        )

            else:
                response = self._get_ai_reply(message_sequence=input_message_sequence)

            # Step 2: check if LLM wanted to call a function
            # (if yes) Step 3: call the function
            # (if yes) Step 4: send the info on the function call and function response to LLM
            response_message = response.choices[0].message
            response_message.model_copy()
            all_response_messages, heartbeat_request, function_failed = (
                self._handle_ai_response(response_message)
            )

            # Step 4: extend the message history
            if user_message is not None:
                if isinstance(user_message, Message):
                    all_new_messages = [user_message] + all_response_messages
                else:
                    raise ValueError(type(user_message))
            else:
                all_new_messages = all_response_messages

            # Check the memory pressure and potentially issue a memory pressure warning
            current_total_tokens = response.usage.total_tokens
            active_memory_warning = False
            # We can't do summarize logic properly if context_window is undefined
            if self.agent_state.llm_config.context_window is None:
                # Fallback if for some reason context_window is missing, just set to the default
                print(
                    f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}"
                )
                print(f"{self.agent_state}")
                self.agent_state.llm_config.context_window = (
                    LLM_MAX_TOKENS[self.model]
                    if (self.model is not None and self.model in LLM_MAX_TOKENS)
                    else LLM_MAX_TOKENS["DEFAULT"]
                )

            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * int(
                self.agent_state.llm_config.context_window
            ):
                print(
                    f"{CLI_WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )
                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = (
                        True  # it's up to the outer loop to handle this
                    )
            else:
                print(
                    f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )

            self._append_to_messages(all_new_messages)
            messages_to_return = (
                [msg.to_openai_dict() for msg in all_new_messages]
                if return_dicts
                else all_new_messages
            )
            return (
                messages_to_return,
                heartbeat_request,
                function_failed,
                active_memory_warning,
                response.usage.completion_tokens,
            )
        except Exception as e:
            print(f"step() failed\nuser_message = {user_message}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                # A separate API call to run a summarizer
                self.summarize_messages_inplace()

                # Try step again
                return self.step(
                    user_message, first_message=first_message, return_dicts=return_dicts
                )
            else:
                print(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e


def save_agent(agent: Agent, ms: MetadataStore):
    """Save agent to metadata store"""

    agent.update_state()
    agent_state = agent.agent_state

    if ms.get_agent(agent_name=agent_state.name, user_id=agent_state.user_id):
        ms.update_agent(agent_state)
    else:
        ms.create_agent(agent_state)
