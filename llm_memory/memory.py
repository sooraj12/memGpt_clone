import uuid

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union
from data_types import Message, AgentState, Passage
from embeddings import embedding_model, parse_and_chunk_text, query_embedding
from utils import get_local_time, count_tokens
from constants import MESSAGE_SUMMARY_WARNING_FRAC, MESSAGE_SUMMARY_REQUEST_ACK
from llm_api.llm_api_tools import create
from prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM


def _format_summary_history(message_history: List[Message]):
    # TODO use existing prompt formatters for this (eg ChatML)
    return "\n".join([f"{m.role}: {m.text}" for m in message_history])


def summarize_messages(
    agent_state: AgentState,
    message_sequence_to_summarize: List[Message],
    insert_acknowledgement_assistant_message: bool = True,
):
    """Summarize a message sequence using GPT"""
    # we need the context_window
    context_window = agent_state.llm_config.context_window

    summary_prompt = SUMMARY_PROMPT_SYSTEM
    summary_input = _format_summary_history(message_sequence_to_summarize)
    summary_input_tkns = count_tokens(summary_input)
    if summary_input_tkns > MESSAGE_SUMMARY_WARNING_FRAC * context_window:
        trunc_ratio = (
            MESSAGE_SUMMARY_WARNING_FRAC * context_window / summary_input_tkns
        ) * 0.8  # For good measure...
        cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
        summary_input = str(
            [
                summarize_messages(
                    agent_state,
                    message_sequence_to_summarize=message_sequence_to_summarize[
                        :cutoff
                    ],
                )
            ]
            + message_sequence_to_summarize[cutoff:]
        )

    dummy_user_id = uuid.uuid4()
    dummy_agent_id = uuid.uuid4()
    message_sequence = []
    message_sequence.append(
        Message(
            user_id=dummy_user_id,
            agent_id=dummy_agent_id,
            role="system",
            text=summary_prompt,
        )
    )
    if insert_acknowledgement_assistant_message:
        message_sequence.append(
            Message(
                user_id=dummy_user_id,
                agent_id=dummy_agent_id,
                role="assistant",
                text=MESSAGE_SUMMARY_REQUEST_ACK,
            )
        )
    message_sequence.append(
        Message(
            user_id=dummy_user_id,
            agent_id=dummy_agent_id,
            role="user",
            text=summary_input,
        )
    )

    response = create(
        llm_config=agent_state.llm_config,
        user_id=agent_state.user_id,
        messages=message_sequence,
    )

    print(f"summarize_messages gpt reply: {response.choices[0]}")
    reply = response.choices[0].message.content
    return reply


class CoreMemory:
    """Held in-context inside the system message

    Core Memory: Refers to the system block, which provides essential, foundational context to the AI.
    This includes the persona information, essential user details,
    and any other baseline data you deem necessary for the AI's basic functioning.
    """

    def __init__(
        self,
        persona=None,
        human=None,
        persona_char_limit=None,
        human_char_limit=None,
        archival_memory_exists=True,
    ):
        self.persona = persona
        self.human = human
        self.persona_char_limit = persona_char_limit
        self.human_char_limit = human_char_limit

        # affects the error message the AI will see on overflow inserts
        self.archival_memory_exists = archival_memory_exists

    def __repr__(self) -> str:
        return (
            "\n### CORE MEMORY ###"
            + f"\n=== Persona ===\n{self.persona}"
            + f"\n\n=== Human ===\n{self.human}"
        )

    @classmethod
    def load(cls, state):
        return cls(state["persona"], state["human"])

    def to_dict(self):
        return {
            "persona": self.persona,
            "human": self.human,
        }

    def edit_persona(self, new_persona):
        if self.persona_char_limit and len(new_persona) > self.persona_char_limit:
            error_msg = f"Edit failed: Exceeds {self.persona_char_limit} character limit (requested {len(new_persona)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'persona' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.persona = new_persona
        return len(self.persona)

    def edit_human(self, new_human):
        if self.human_char_limit and len(new_human) > self.human_char_limit:
            error_msg = f"Edit failed: Exceeds {self.human_char_limit} character limit (requested {len(new_human)})."
            if self.archival_memory_exists:
                error_msg = f"{error_msg} Consider summarizing existing core memories in 'human' and/or moving lower priority content to archival memory to free up space in core memory, then trying again."
            raise ValueError(error_msg)

        self.human = new_human
        return len(self.human)

    def edit(self, field, content):
        if field == "persona":
            return self.edit_persona(content)
        elif field == "human":
            return self.edit_human(content)
        else:
            raise KeyError(
                f'No memory section named {field} (must be either "persona" or "human")'
            )

    def edit_append(self, field, content, sep="\n"):
        if field == "persona":
            new_content = self.persona + sep + content
            return self.edit_persona(new_content)
        elif field == "human":
            new_content = self.human + sep + content
            return self.edit_human(new_content)
        else:
            raise KeyError(
                f'No memory section named {field} (must be either "persona" or "human")'
            )

    def edit_replace(self, field, old_content, new_content):
        if len(old_content) == 0:
            raise ValueError(
                "old_content cannot be an empty string (must specify old_content to replace)"
            )

        if field == "persona":
            if old_content in self.persona:
                new_persona = self.persona.replace(old_content, new_content)
                return self.edit_persona(new_persona)
            else:
                raise ValueError(
                    "Content not found in persona (make sure to use exact string)"
                )
        elif field == "human":
            if old_content in self.human:
                new_human = self.human.replace(old_content, new_content)
                return self.edit_human(new_human)
            else:
                raise ValueError(
                    "Content not found in human (make sure to use exact string)"
                )
        else:
            raise KeyError(
                f'No memory section named {field} (must be either "persona" or "human")'
            )


class ArchivalMemory(ABC):
    @abstractmethod
    def insert(self, memory_string: str):
        """Insert new archival memory

        :param memory_string: Memory string to insert
        :type memory_string: str
        """

    @abstractmethod
    def search(self, query_string, count=None, start=None) -> Tuple[List[str], int]:
        """Search archival memory

        :param query_string: Query string
        :type query_string: str
        :param count: Number of results to return (None for all)
        :type count: Optional[int]
        :param start: Offset to start returning results from (None if 0)
        :type start: Optional[int]

        :return: Tuple of (list of results, total number of results)
        """

    @abstractmethod
    def __repr__(self) -> str:
        pass


class RecallMemory(ABC):
    @abstractmethod
    def text_search(self, query_string, count=None, start=None):
        """Search messages that match query_string in recall memory"""

    @abstractmethod
    def date_search(self, start_date, end_date, count=None, start=None):
        """Search messages between start_date and end_date in recall memory"""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def insert(self, message: Message):
        """Insert message into recall memory"""


class BaseRecallMemory(RecallMemory):
    """Recall memory based on base functions implemented by storage connectors"""

    def __init__(self, agent_state, restrict_search_to_summaries=False):
        # If true, the pool of messages that can be queried are the automated summaries only
        # (generated when the conversation window needs to be shortened)
        self.restrict_search_to_summaries = restrict_search_to_summaries
        from agent_store.storage import StorageConnector

        self.agent_state = agent_state

        # create embedding model
        self.embed_model = embedding_model(agent_state.embedding_config)
        self.embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size

        # create storage backend
        self.storage = StorageConnector.get_recall_storage_connector(
            user_id=agent_state.user_id, agent_id=agent_state.id
        )
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def get_all(self, start=0, count=None):
        results = self.storage.get_all(start, count)
        results_json = [message.to_openai_dict() for message in results]
        return results_json, len(results)

    def text_search(self, query_string, count=None, start=None):
        results = self.storage.query_text(query_string, count, start)
        results_json = [message.to_openai_dict() for message in results]
        return results_json, len(results)

    def date_search(self, start_date, end_date, count=None, start=None):
        results = self.storage.query_date(start_date, end_date, count, start)
        results_json = [message.to_openai_dict() for message in results]
        return results_json, len(results)

    def __repr__(self) -> str:
        total = self.storage.size()
        system_count = self.storage.size(filters={"role": "system"})
        user_count = self.storage.size(filters={"role": "user"})
        assistant_count = self.storage.size(filters={"role": "assistant"})
        function_count = self.storage.size(filters={"role": "function"})
        other_count = total - (
            system_count + user_count + assistant_count + function_count
        )

        memory_str = (
            "Statistics:"
            + f"\n{total} total messages"
            + f"\n{system_count} system"
            + f"\n{user_count} user"
            + f"\n{assistant_count} assistant"
            + f"\n{function_count} function"
            + f"\n{other_count} other"
        )
        return "\n### RECALL MEMORY ###" + f"\n{memory_str}"

    def insert(self, message: Message):
        self.storage.insert(message)

    def insert_many(self, messages: List[Message]):
        self.storage.insert_many(messages)

    def save(self):
        self.storage.save()

    def __len__(self):
        return self.storage.size()


class EmbeddingArchivalMemory(ArchivalMemory):
    """Archival memory with embedding based search"""

    def __init__(self, agent_state: AgentState, top_k: Optional[int] = 100):
        """Init function for archival memory

        :param archival_memory_database: name of dataset to pre-fill archival with
        :type archival_memory_database: str
        """
        from agent_store.storage import StorageConnector

        self.top_k = top_k
        self.agent_state = agent_state

        # create embedding model
        self.embed_model = embedding_model(agent_state.embedding_config)
        self.embedding_chunk_size = agent_state.embedding_config.embedding_chunk_size
        assert (
            self.embedding_chunk_size
        ), f"Must set {agent_state.embedding_config.embedding_chunk_size}"

        # create storage backend
        self.storage = StorageConnector.get_archival_storage_connector(
            user_id=agent_state.user_id, agent_id=agent_state.id
        )
        # TODO: have some mechanism for cleanup otherwise will lead to OOM
        self.cache = {}

    def create_passage(self, text, embedding):
        return Passage(
            user_id=self.agent_state.user_id,
            agent_id=self.agent_state.id,
            text=text,
            embedding=embedding,
            embedding_dim=self.agent_state.embedding_config.embedding_dim,
            embedding_model=self.agent_state.embedding_config.embedding_model,
        )

    def save(self):
        """Save the index to disk"""
        self.storage.save()

    def insert(self, memory_string, return_ids=False) -> Union[bool, List[uuid.UUID]]:
        """Embed and save memory string"""

        if not isinstance(memory_string, str):
            raise TypeError("memory must be a string")

        try:
            passages = []

            # breakup string into passages
            for text in parse_and_chunk_text(memory_string, self.embedding_chunk_size):
                embedding = self.embed_model.get_text_embedding(text)
                # fixing weird bug where type returned isn't a list, but instead is an object
                # eg: embedding={'object': 'list', 'data': [{'object': 'embedding', 'embedding': [-0.0071973633, -0.07893023,
                if isinstance(embedding, dict):
                    try:
                        embedding = embedding["data"][0]["embedding"]
                    except (KeyError, IndexError):
                        # TODO as a fallback, see if we can find any lists in the payload
                        raise TypeError(
                            f"Got back an unexpected payload from text embedding function, type={type(embedding)}, value={embedding}"
                        )
                passages.append(self.create_passage(text, embedding))

            # grab the return IDs before the list gets modified
            ids = [str(p.id) for p in passages]

            # insert passages
            self.storage.insert_many(passages)

            if return_ids:
                return ids
            else:
                return True

        except Exception as e:
            print("Archival insert error", e)
            raise e

    def search(self, query_string, count=None, start=None):
        """Search query string"""
        if not isinstance(query_string, str):
            return TypeError("query must be a string")

        try:
            if query_string not in self.cache:
                # self.cache[query_string] = self.retriever.retrieve(query_string)
                query_vec = query_embedding(self.embed_model, query_string)
                self.cache[query_string] = self.storage.query(
                    query_string, query_vec, top_k=self.top_k
                )

            start = int(start if start else 0)
            count = int(count if count else self.top_k)
            end = min(count + start, len(self.cache[query_string]))

            results = self.cache[query_string][start:end]
            results = [
                {"timestamp": get_local_time(), "content": node.text}
                for node in results
            ]
            return results, len(results)
        except Exception as e:
            print("Archival search error", e)
            raise e

    def __repr__(self) -> str:
        limit = 10
        passages = []
        for passage in list(
            self.storage.get_all(limit=limit)
        ):  # TODO: only get first 10
            passages.append(str(passage.text))
        memory_str = "\n".join(passages)
        return (
            "\n### ARCHIVAL MEMORY ###"
            + f"\n{memory_str}"
            + f"\nSize: {self.storage.size()}"
        )

    def __len__(self):
        return self.storage.size()
