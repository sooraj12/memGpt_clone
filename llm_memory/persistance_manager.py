from memory import BaseRecallMemory, EmbeddingArchivalMemory
from data_types import AgentState, Message
from typing import List


class LocalStateManager:
    """In-memory state manager has nothing to manage, all agents are held in-memory"""

    recall_memory_cls = BaseRecallMemory
    archival_memory_cls = EmbeddingArchivalMemory

    def __init__(self, agent_state: AgentState):
        # Memory held in-state useful for debugging stateful versions
        self.memory = None
        # self.messages = []  # current in-context messages
        # self.all_messages = [] # all messages seen in current session (needed if lazily synchronizing state with DB)
        self.archival_memory = EmbeddingArchivalMemory(agent_state)
        self.recall_memory = BaseRecallMemory(agent_state)
        # self.agent_state = agent_state

    def save(self):
        """Ensure storage connectors save data"""
        self.archival_memory.save()
        self.recall_memory.save()

    def init(self, agent):
        """Connect persistent state manager to agent"""
        print(f"Initializing {self.__class__.__name__} with agent object")
        # self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        # self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        # print(f"{self.__class__.__name__}.all_messages.len = {len(self.all_messages)}")
        print(f"{self.__class__.__name__}.messages.len = {len(self.messages)}")

    def trim_messages(self, num):
        # print(f"InMemoryStateManager.trim_messages")
        # self.messages = [self.messages[0]] + self.messages[num:]
        pass

    def prepend_to_messages(self, added_messages: List[Message]):
        # first tag with timestamps
        # added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        print(f"{self.__class__.__name__}.prepend_to_message")
        # self.messages = [self.messages[0]] + added_messages + self.messages[1:]

        # add to recall memory
        self.recall_memory.insert_many([m for m in added_messages])

    def append_to_messages(self, added_messages: List[Message]):
        # first tag with timestamps
        # added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        print(f"{self.__class__.__name__}.append_to_messages")
        # self.messages = self.messages + added_messages

        # add to recall memory
        self.recall_memory.insert_many([m for m in added_messages])

    def swap_system_message(self, new_system_message: Message):
        # first tag with timestamps
        # new_system_message = {"timestamp": get_local_time(), "message": new_system_message}

        print(f"{self.__class__.__name__}.swap_system_message")
        # self.messages[0] = new_system_message

        # add to recall memory
        self.recall_memory.insert(new_system_message)

    def update_memory(self, new_memory):
        print(f"{self.__class__.__name__}.update_memory")
        self.memory = new_memory
