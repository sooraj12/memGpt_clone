import uuid
import os
import configparser

from dataclasses import dataclass
from typing import Optional
from constants import (
    DEFAULT_HUMAN,
    DEFAULT_PERSONA,
    DEFAULT_PRESET,
    MEMGPT_DIR,
    LLM_MAX_TOKENS,
)


class EmbeddingConfig:
    def __init__(
        self,
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_chunk_size: Optional[int] = 300,
    ):
        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_chunk_size = embedding_chunk_size


class LLMConfig:
    def __init__(
        self,
        model: Optional[str] = None,
        model_endpoint: Optional[str] = None,
        model_wrapper: Optional[str] = None,
        context_window: Optional[int] = None,
    ):
        self.model = model
        self.model_endpoint = model_endpoint
        self.model_wrapper = model_wrapper
        self.context_window = context_window

        if context_window is None:
            self.context_window = (
                LLM_MAX_TOKENS[self.model]
                if self.model in LLM_MAX_TOKENS
                else LLM_MAX_TOKENS["DEFAULT"]
            )
        else:
            self.context_window = context_window


def get_field(config, section, field):
    if section not in config:
        return None
    if config.has_option(section, field):
        return config.get(section, field)
    else:
        return None


def set_field(config, section, field, value):
    if value is None:  # cannot write None
        return
    if section not in config:  # create section
        config.add_section(section)
    config.set(section, field, value)


@dataclass
class MemGPTConfig:
    config_path: str = os.path.join(MEMGPT_DIR, "config")
    anon_clientid: str = str(uuid.UUID(int=0))

    # preset
    preset: str = DEFAULT_PRESET

    # persona parameters
    persona: str = DEFAULT_PERSONA
    human: str = DEFAULT_HUMAN

    # model parameters
    default_llm_config: LLMConfig = None

    # embedding parameters
    default_embedding_config: EmbeddingConfig = None

    # database configs: archival
    archival_storage_type: str = "chroma"  # local, db
    archival_storage_path: str = os.path.join(MEMGPT_DIR, "chroma")
    archival_storage_uri: str = None

    # database configs: recall
    recall_storage_type: str = "sqlite"  # local, db
    recall_storage_path: str = MEMGPT_DIR
    recall_storage_uri: str = None  # TODO: eventually allow external vector DB

    # database configs: metadata storage (sources, agents, data sources)
    metadata_storage_type: str = "sqlite"
    metadata_storage_path: str = MEMGPT_DIR
    metadata_storage_uri: str = None

    def save(self):
        config = configparser.ConfigParser()

        # CLI defaults
        set_field(config, "defaults", "preset", self.preset)
        set_field(config, "defaults", "persona", self.persona)
        set_field(config, "defaults", "human", self.human)

        # model defaults
        set_field(config, "model", "model", self.default_llm_config.model)
        set_field(
            config, "model", "model_endpoint", self.default_llm_config.model_endpoint
        )
        set_field(
            config, "model", "model_wrapper", self.default_llm_config.model_wrapper
        )
        set_field(
            config,
            "model",
            "context_window",
            str(self.default_llm_config.context_window),
        )

        # embeddings
        set_field(
            config,
            "embedding",
            "embedding_endpoint",
            self.default_embedding_config.embedding_endpoint,
        )
        set_field(
            config,
            "embedding",
            "embedding_model",
            self.default_embedding_config.embedding_model,
        )
        set_field(
            config,
            "embedding",
            "embedding_dim",
            str(self.default_embedding_config.embedding_dim),
        )
        set_field(
            config,
            "embedding",
            "embedding_chunk_size",
            str(self.default_embedding_config.embedding_chunk_size),
        )

        # archival storage
        set_field(config, "archival_storage", "type", self.archival_storage_type)
        set_field(config, "archival_storage", "path", self.archival_storage_path)

        # recall storage
        set_field(config, "recall_storage", "type", self.recall_storage_type)
        set_field(config, "recall_storage", "path", self.recall_storage_path)
        set_field(config, "recall_storage", "uri", self.recall_storage_uri)

        # metadata storage
        set_field(config, "metadata_storage", "type", self.metadata_storage_type)
        set_field(config, "metadata_storage", "path", self.metadata_storage_path)
        set_field(config, "metadata_storage", "uri", self.metadata_storage_uri)

        # client
        if not self.anon_clientid:
            self.anon_clientid = self.generate_uuid()
        set_field(config, "client", "anon_clientid", self.anon_clientid)

        # always make sure all directories are present
        self.create_config_dir()

        with open(self.config_path, "w", encoding="utf-8") as f:
            config.write(f)

    @staticmethod
    def generate_uuid() -> str:
        return uuid.UUID(int=uuid.getnode()).hex

    @staticmethod
    def exists():
        config_path = MemGPTConfig.config_path
        return os.path.exists(config_path)

    @staticmethod
    def create_config_dir():
        if not os.path.exists(MEMGPT_DIR):
            os.makedirs(MEMGPT_DIR, exist_ok=True)

        folders = [
            "personas",
            "humans",
            "archival",
            "agents",
            "functions",
            "system_prompts",
            "presets",
            "settings",
        ]

        for folder in folders:
            if not os.path.exists(os.path.join(MEMGPT_DIR, folder)):
                os.makedirs(os.path.join(MEMGPT_DIR, folder))

    @classmethod
    def load(cls):
        config = configparser.ConfigParser()
        config_path = MemGPTConfig.config_path

        cls.create_config_dir()

        # load existing
        if os.path.exists(config_path):
            config.read(config_path)

            llm_config_dict = {
                # Extract relevant LLM configuration from the config file
                "model": get_field(config, "model", "model"),
                "model_endpoint": get_field(config, "model", "model_endpoint"),
                "model_wrapper": get_field(config, "model", "model_wrapper"),
                "context_window": get_field(config, "model", "context_window"),
            }
            embedding_config_dict = {
                # Extract relevant Embedding configuration from the config file
                "embedding_endpoint": get_field(
                    config, "embedding", "embedding_endpoint"
                ),
                "embedding_model": get_field(config, "embedding", "embedding_model"),
                "embedding_dim": get_field(config, "embedding", "embedding_dim"),
                "embedding_chunk_size": get_field(
                    config, "embedding", "embedding_chunk_size"
                ),
            }

            # Remove null values
            llm_config_dict = {
                k: v for k, v in llm_config_dict.items() if v is not None
            }
            embedding_config_dict = {
                k: v for k, v in embedding_config_dict.items() if v is not None
            }

            # Correct the types that aren't strings
            if llm_config_dict["context_window"] is not None:
                llm_config_dict["context_window"] = int(
                    llm_config_dict["context_window"]
                )
            if embedding_config_dict["embedding_dim"] is not None:
                embedding_config_dict["embedding_dim"] = int(
                    embedding_config_dict["embedding_dim"]
                )
            if embedding_config_dict["embedding_chunk_size"] is not None:
                embedding_config_dict["embedding_chunk_size"] = int(
                    embedding_config_dict["embedding_chunk_size"]
                )
            # Construct the inner properties
            llm_config = LLMConfig(**llm_config_dict)
            embedding_config = EmbeddingConfig(**embedding_config_dict)

            config_dict = {
                # Two prepared configs
                "default_llm_config": llm_config,
                "default_embedding_config": embedding_config,
                # Agent related
                "preset": get_field(config, "defaults", "preset"),
                "persona": get_field(config, "defaults", "persona"),
                "human": get_field(config, "defaults", "human"),
                "agent": get_field(config, "defaults", "agent"),
                # Storage related
                "archival_storage_type": get_field(config, "archival_storage", "type"),
                "archival_storage_path": get_field(config, "archival_storage", "path"),
                "recall_storage_type": get_field(config, "recall_storage", "type"),
                "recall_storage_uri": get_field(config, "recall_storage", "uri"),
                "metadata_storage_type": get_field(config, "metadata_storage", "type"),
                "metadata_storage_path": get_field(config, "metadata_storage", "path"),
                "metadata_storage_uri": get_field(config, "metadata_storage", "uri"),
                # Misc
                "anon_clientid": get_field(config, "client", "anon_clientid"),
                "config_path": config_path,
            }

            # Don't include null values
            config_dict = {k: v for k, v in config_dict.items() if v is not None}

            return cls(**config_dict)

        # create new config
        anon_clientid = MemGPTConfig.generate_uuid()
        config = cls(anon_clientid=anon_clientid, config_path=config_path)
        config.create_config_dir()

        return config
