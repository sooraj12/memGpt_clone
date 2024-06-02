import uuid
import system
import logging
from abc import abstractmethod

from interface import QueuingInterface
from typing import Callable
from functools import wraps
from threading import Lock
from fastapi import HTTPException
from config import MemGPTConfig, LLMConfig, EmbeddingConfig
from metadata import MetadataStore
from data_types import User, Message
from presets import presets
from typing import Optional, Union
from datetime import datetime
from agent import Agent, save_agent
from pprint import pprint
from constants import REQ_HEARTBEAT_MESSAGE, FUNC_FAILED_HEARTBEAT_MESSAGE
from data_types import AgentState

logger = logging.getLogger(__name__)


class Server(object):
    """Abstract server class that supports multi-agent multi-user"""

    @abstractmethod
    def list_agents(self, user_id: uuid.UUID) -> dict:
        """List all available agents to a user"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_messages(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, start: int, count: int
    ) -> list:
        """Paginated query of in-context messages in agent message queue"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_memory(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> dict:
        """Return the memory of an agent (core memory + non-core statistics)"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_config(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> dict:
        """Return the config of an agent"""
        raise NotImplementedError

    @abstractmethod
    def get_server_config(self, user_id: uuid.UUID) -> dict:
        """Return the base config"""
        raise NotImplementedError

    @abstractmethod
    def update_agent_core_memory(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, new_memory_contents: dict
    ) -> dict:
        """Update the agents core memory block, return the new state"""
        raise NotImplementedError

    @abstractmethod
    def create_agent(
        self,
        user_id: uuid.UUID,
        agent_config: Union[dict, AgentState],
        interface: Union[QueuingInterface, None],
        # persistence_manager: Union[PersistenceManager, None],
    ) -> str:
        """Create a new agent using a config"""
        raise NotImplementedError

    @abstractmethod
    def user_message(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str
    ) -> None:
        """Process a message from the user, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def system_message(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str
    ) -> None:
        """Process a message from the system, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def run_command(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, command: str
    ) -> Union[str, None]:
        """Run a command on the agent, e.g. /memory

        May return a string with a message generated by the command
        """
        raise NotImplementedError


class LockingServer(Server):
    """Basic support for concurrency protections (all requests that modify an agent lock the agent until the operation is complete)"""

    # Locks for each agent
    _agent_locks = {}

    @staticmethod
    def agent_lock_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, user_id: uuid.UUID, agent_id: uuid.UUID, *args, **kwargs):
            # logger.info("Locking check")

            # Initialize the lock for the agent_id if it doesn't exist
            if agent_id not in self._agent_locks:
                # logger.info(f"Creating lock for agent_id = {agent_id}")
                self._agent_locks[agent_id] = Lock()

            # Check if the agent is currently locked
            if not self._agent_locks[agent_id].acquire(blocking=False):
                # logger.info(f"agent_id = {agent_id} is busy")
                raise HTTPException(
                    status_code=423, detail=f"Agent '{agent_id}' is currently busy."
                )

            try:
                # Execute the function
                # logger.info(f"running function on agent_id = {agent_id}")
                print("USERID", user_id)
                return func(self, user_id, agent_id, *args, **kwargs)
            finally:
                # Release the lock
                # logger.info(f"releasing lock on agent_id = {agent_id}")
                self._agent_locks[agent_id].release()

        return wrapper

    @agent_lock_decorator
    def user_message(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, message: str
    ) -> None:
        raise NotImplementedError

    @agent_lock_decorator
    def run_command(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, command: str
    ) -> Union[str, None]:
        raise NotImplementedError


class SyncServer(LockingServer):
    default_agent = "Memgpt_agent"

    def __init__(
        self,
        default_interface: QueuingInterface,
        chaining: bool = True,
        max_chaining_steps: bool = None,
    ):
        self.active_agents = []
        self.chaining = chaining
        self.max_chaining_steps = max_chaining_steps
        self.default_interface = default_interface

        # initialize the connection to db
        self.config = MemGPTConfig.load()
        print(f"server :: loading configuration from '{self.config.config_path}'")

        # generate default LLM/Embedding configs for the server
        self.server_llm_config = LLMConfig(
            model=self.config.default_llm_config.model,
            model_endpoint=self.config.default_llm_config.model_endpoint,
            model_wrapper=self.config.default_llm_config.model_wrapper,
            context_window=self.config.default_llm_config.context_window,
        )
        self.server_embedding_config = EmbeddingConfig(
            embedding_endpoint=self.config.default_embedding_config.embedding_endpoint,
            embedding_dim=self.config.default_embedding_config.embedding_dim,
            embedding_model=self.config.default_embedding_config.embedding_model,
            embedding_chunk_size=self.config.default_embedding_config.embedding_chunk_size,
        )

        # initialize the metadata store
        self.ms = MetadataStore(self.config)

        # pre-fill the database(users, presets, humans, personas, agent)
        # pre-fill agents
        user_id = uuid.UUID(self.config.anon_clientid)

        presets.add_default_presets(user_id, self.ms)
        agents = self.ms.list_agents(user_id=user_id)
        if len(agents) == 0:
            # create a new agent
            agent_name = self.default_agent
            llm_config = self.config.default_llm_config
            embedding_config = self.config.default_embedding_config
            preset_obj = self.ms.get_preset(name=self.config.preset, user_id=user_id)

            memgpt_agent = Agent(
                interface=self.default_interface,
                name=agent_name,
                created_by=user_id,
                preset=preset_obj,
                llm_config=llm_config,
                embedding_config=embedding_config,
                first_message_verify_mono=False,
            )

            save_agent(agent=memgpt_agent, ms=self.ms)

        user = User(
            id=uuid.UUID(self.config.anon_clientid),
        )

        if self.ms.get_user(user_id):
            # update user
            self.ms.update_user(user)
        else:
            self.ms.create_user(user)

    def _add_agent(
        self, user_id: uuid.UUID, agent_id: uuid.UUID, agent_obj: Agent
    ) -> None:
        """Put an agent object inside the in-memory object store"""
        # Make sure the agent doesn't already exist
        if self._get_agent(user_id=user_id, agent_id=agent_id) is not None:
            # Can be triggered on concurrent request, so don't throw a full error
            # raise KeyError(f"Agent (user={user_id}, agent={agent_id}) is already loaded")
            logger.exception(
                f"Agent (user={user_id}, agent={agent_id}) is already loaded"
            )
            return
        # Add Agent instance to the in-memory list
        self.active_agents.append(
            {
                "user_id": str(user_id),
                "agent_id": str(agent_id),
                "agent": agent_obj,
            }
        )

    def _get_agent(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> Union[Agent, None]:
        """Get the agent object from the in-memory object store"""
        for d in self.active_agents:
            if d["user_id"] == str(user_id) and d["agent_id"] == str(agent_id):
                return d["agent"]
        return None

    def _load_agent(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        interface: Union[QueuingInterface, None] = None,
    ) -> Agent:
        """Loads a saved agent into memory (if it doesn't exist, throw an error)"""
        assert isinstance(user_id, uuid.UUID), user_id
        assert isinstance(agent_id, uuid.UUID), agent_id

        # If an interface isn't specified, use the default
        if interface is None:
            interface = self.default_interface

        try:
            logger.info(
                f"Grabbing agent user_id={user_id} agent_id={agent_id} from database"
            )
            agent_state = self.ms.get_agent(agent_id=agent_id, user_id=user_id)
            if not agent_state:
                logger.exception(f"agent_id {agent_id} does not exist")
                raise ValueError(f"agent_id {agent_id} does not exist")

            # Instantiate an agent object using the state retrieved
            logger.info("Creating an agent object")
            pprint(agent_state)
            memgpt_agent = Agent(agent_state=agent_state, interface=interface)

            # Add the agent to the in-memory store and return its reference
            logger.info(
                f"Adding agent to the agent cache: user_id={user_id}, agent_id={agent_id}"
            )
            self._add_agent(user_id=user_id, agent_id=agent_id, agent_obj=memgpt_agent)
            return memgpt_agent

        except Exception as e:
            logger.exception(
                f"Error occurred while trying to get agent {agent_id}:\n{e}"
            )
            raise

    def _get_or_load_agent(self, user_id: uuid.UUID, agent_id: uuid.UUID) -> Agent:
        """Check if the agent is in-memory, then load"""
        logger.info(f"Checking for agent user_id={user_id} agent_id={agent_id}")
        memgpt_agent = self._get_agent(user_id=user_id, agent_id=agent_id)
        if not memgpt_agent:
            logger.info(
                f"Agent not loaded, loading agent user_id={user_id} agent_id={agent_id}"
            )
            memgpt_agent = self._load_agent(user_id=user_id, agent_id=agent_id)
        return memgpt_agent

    def _step(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        input_message: Union[str, Message],
    ) -> int:
        """Send the input message through the agent"""

        print(f"Got input message: {input_message}")

        # Get the agent object (loaded in memory)
        memgpt_agent = self._get_or_load_agent(user_id=user_id, agent_id=agent_id)
        if memgpt_agent is None:
            raise KeyError(f"Agent (user={user_id}, agent={agent_id}) is not loaded")

        print("Starting agent step")

        no_verify = True
        next_input_message = input_message
        counter = 0
        while True:
            (
                new_messages,
                heartbeat_request,
                function_failed,
                token_warning,
                tokens_accumulated,
            ) = memgpt_agent.step(
                next_input_message,
                first_message=False,
                skip_verify=no_verify,
                return_dicts=False,
            )
            counter += 1

            # Chain stops
            if not self.chaining:
                logger.debug("No chaining, stopping after one step")
                break
            elif (
                self.max_chaining_steps is not None
                and counter > self.max_chaining_steps
            ):
                logger.debug(f"Hit max chaining steps, stopping after {counter} steps")
                break
            # Chain handlers
            elif token_warning:
                next_input_message = system.get_token_limit_warning()
                continue  # always chain
            elif function_failed:
                next_input_message = system.get_heartbeat(FUNC_FAILED_HEARTBEAT_MESSAGE)
                continue  # always chain
            elif heartbeat_request:
                next_input_message = system.get_heartbeat(REQ_HEARTBEAT_MESSAGE)
                continue  # always chain
            # MemGPT no-op / yield
            else:
                break

        print("Finished agent step")
        memgpt_agent.interface.step_yield()

        return tokens_accumulated

    @LockingServer.agent_lock_decorator
    def user_message(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        message: Union[str, Message],
    ):
        """Process an incoming user message and feed it through the MemGPT agent"""
        if self.ms.get_user(user_id=user_id) is None:
            raise ValueError(f"User user_id={user_id} does not exist")
        if self.ms.get_agent(agent_id=agent_id, user_id=user_id) is None:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Basic input sanitization
        if isinstance(message, str):
            if len(message) == 0:
                raise ValueError(f"Invalid input: '{message}'")

            # If the input begins with a command prefix, reject
            elif message.startswith("/"):
                raise ValueError(f"Invalid input: '{message}'")

            packaged_user_message = system.package_user_message(user_message=message)

            message = Message(
                user_id=user_id,
                agent_id=agent_id,
                role="user",
                text=packaged_user_message,
            )

        if isinstance(message, Message):
            # Can't have a null text field
            if len(message.text) == 0 or message.text is None:
                raise ValueError(f"Invalid input: '{message.text}'")
            # If the input begins with a command prefix, reject
            elif message.text.startswith("/"):
                raise ValueError(f"Invalid input: '{message.text}'")

        else:
            raise TypeError(f"Invalid input: '{message}' - type {type(message)}")

        # Run the agent state forward
        self._step(
            user_id=user_id, agent_id=agent_id, input_message=packaged_user_message
        )

    @LockingServer.agent_lock_decorator
    def system_message(
        self,
        user_id: uuid.UUID,
        agent_id: uuid.UUID,
        message: Union[str, Message],
        timestamp: Optional[datetime] = None,
    ):
        pass

    def authenticate_user(self) -> uuid.UUID:
        return uuid.UUID(MemGPTConfig.load().anon_clientid)

    def api_key_to_user(self, api_key: str) -> uuid.UUID:
        """Decode an API key to a user"""
        user = self.ms.get_user_from_api_key(api_key=api_key)
        print("got user", api_key, user.id)
        if user is None:
            raise HTTPException(status_code=403, detail="Invalid credentials")
        else:
            return user.id
