import uuid

from config import MemGPTConfig
from sqlalchemy import (
    create_engine,
    Column,
    String,
    CHAR,
    TypeDecorator,
    DateTime,
    func,
    JSON,
    BIGINT,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from data_types import User, AgentState, Source, Token, Preset
from config import LLMConfig, EmbeddingConfig
from models.pydantic_models import HumanModel, PersonaModel, ToolModel, JobModel
from utils import enforce_types
from typing import Optional, List

Base = declarative_base()


class CommonUUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR())

    def process_bind_param(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return str(value)  # Convert UUID to string for SQLite

    def process_result_value(self, value, dialect):
        if dialect.name == "postgresql" or value is None:
            return value
        else:
            return uuid.UUID(value)


class UserModel(Base):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    # name = Column(String, nullable=False)
    default_agent = Column(String)

    def __repr__(self) -> str:
        return f"<User(id='{self.id}')>"

    def to_record(self) -> User:
        return User(
            id=self.id,
            # name=self.name
            default_agent=self.default_agent,
        )


class LLMConfigColumn(TypeDecorator):
    """Custom type for storing LLMConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return vars(value)
        return value

    def process_result_value(self, value, dialect):
        if value:
            return LLMConfig(**value)
        return value


class EmbeddingConfigColumn(TypeDecorator):
    """Custom type for storing EmbeddingConfig as JSON"""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value:
            return vars(value)
        return value

    def process_result_value(self, value, dialect):
        if value:
            return EmbeddingConfig(**value)
        return value


class AgentModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "agents"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    name = Column(String, nullable=False)
    persona = Column(String)
    human = Column(String)
    preset = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # configs
    llm_config = Column(LLMConfigColumn)
    embedding_config = Column(EmbeddingConfigColumn)

    # state
    state = Column(JSON)

    def __repr__(self) -> str:
        return f"<Agent(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> AgentState:
        return AgentState(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            persona=self.persona,
            human=self.human,
            preset=self.preset,
            created_at=self.created_at,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            state=self.state,
        )


class SourceModel(Base):
    """Defines data model for storing Passages (consisting of text, embedding)"""

    __tablename__ = "sources"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    embedding_dim = Column(BIGINT)
    embedding_model = Column(String)
    description = Column(String)

    # TODO: add num passages

    def __repr__(self) -> str:
        return f"<Source(passage_id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Source:
        return Source(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            created_at=self.created_at,
            embedding_dim=self.embedding_dim,
            embedding_model=self.embedding_model,
            description=self.description,
        )


class AgentSourceMappingModel(Base):
    """Stores mapping between agent -> source"""

    __tablename__ = "agent_source_mapping"

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    agent_id = Column(CommonUUID, nullable=False)
    source_id = Column(CommonUUID, nullable=False)

    def __repr__(self) -> str:
        return f"<AgentSourceMapping(user_id='{self.user_id}', agent_id='{self.agent_id}', source_id='{self.source_id}')>"


class TokenModel(Base):
    """Data model for authentication tokens. One-to-many relationship with UserModel (1 User - N tokens)."""

    __tablename__ = "tokens"

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    # each api key is tied to a user account (that it validates access for)
    user_id = Column(CommonUUID, nullable=False)
    # the api key
    token = Column(String, nullable=False)
    # extra (optional) metadata
    name = Column(String)

    def __repr__(self) -> str:
        return f"<Token(id='{self.id}', token='{self.token}', name='{self.name}')>"

    def to_record(self) -> User:
        return Token(
            id=self.id,
            user_id=self.user_id,
            token=self.token,
            name=self.name,
        )


class PresetModel(Base):
    """Defines data model for storing Preset objects"""

    __tablename__ = "presets"
    __table_args__ = {"extend_existing": True}

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    system = Column(String)
    human = Column(String)
    human_name = Column(String, nullable=False)
    persona = Column(String)
    persona_name = Column(String, nullable=False)
    preset = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    functions_schema = Column(JSON)

    def __repr__(self) -> str:
        return f"<Preset(id='{self.id}', name='{self.name}')>"

    def to_record(self) -> Preset:
        return Preset(
            id=self.id,
            user_id=self.user_id,
            name=self.name,
            description=self.description,
            system=self.system,
            human=self.human,
            persona=self.persona,
            human_name=self.human_name,
            persona_name=self.persona_name,
            preset=self.preset,
            created_at=self.created_at,
            functions_schema=self.functions_schema,
        )


class PresetSourceMapping(Base):
    __tablename__ = "preset_source_mapping"

    id = Column(CommonUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(CommonUUID, nullable=False)
    preset_id = Column(CommonUUID, nullable=False)
    source_id = Column(CommonUUID, nullable=False)

    def __repr__(self) -> str:
        return f"<PresetSourceMapping(user_id='{self.user_id}', preset_id='{self.preset_id}', source_id='{self.source_id}')>"


class MetadataStore:
    def __init__(self, config: MemGPTConfig):
        self.uri = config.metadata_storage_uri
        self.engine = create_engine(self.uri)

        try:
            Base.metadata.create_all(
                self.engine,
                tables=[
                    UserModel.__table__,
                    AgentModel.__table__,
                    SourceModel.__table__,
                    AgentSourceMappingModel.__table__,
                    TokenModel.__table__,
                    PresetModel.__table__,
                    PresetSourceMapping.__table__,
                    HumanModel.__table__,
                    PersonaModel.__table__,
                    ToolModel.__table__,
                    JobModel.__table__,
                ],
            )
        except Exception as e:
            print(e)

        self.session_maker = sessionmaker(bind=self.engine)

    @enforce_types
    def get_user(self, user_id: uuid.UUID) -> Optional[User]:
        with self.session_maker() as session:
            results = session.query(UserModel).filter(UserModel.id == user_id).all()
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def create_user(self, user: User):
        with self.session_maker() as session:
            if session.query(UserModel).filter(UserModel.id == user.id).count() > 0:
                raise ValueError(f"User with id {user.id} already exists")
            session.add(UserModel(**vars(user)))
            session.commit()

    @enforce_types
    def update_user(self, user: User):
        with self.session_maker() as session:
            session.query(UserModel).filter(UserModel.id == user.id).update(vars(user))
            session.commit()

    @enforce_types
    def get_preset(
        self,
        preset_id: Optional[uuid.UUID] = None,
        name: Optional[str] = None,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[Preset]:
        with self.session_maker() as session:
            if preset_id:
                results = (
                    session.query(PresetModel).filter(PresetModel.id == preset_id).all()
                )
            elif name and user_id:
                results = (
                    session.query(PresetModel)
                    .filter(PresetModel.name == name)
                    .filter(PresetModel.user_id == user_id)
                    .all()
                )
            else:
                raise ValueError(
                    "Must provide either preset_id or (preset_name and user_id)"
                )
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0].to_record()

    @enforce_types
    def create_preset(self, preset: Preset):
        with self.session_maker() as session:
            if (
                session.query(PresetModel).filter(PresetModel.id == preset.id).count()
                > 0
            ):
                raise ValueError(f"User with id {preset.id} already exists")
            session.add(PresetModel(**vars(preset)))
            session.commit()

    @enforce_types
    def get_persona(self, name: str, user_id: uuid.UUID) -> Optional[PersonaModel]:
        with self.session_maker() as session:
            results = (
                session.query(PersonaModel)
                .filter(PersonaModel.name == name)
                .filter(PersonaModel.user_id == user_id)
                .all()
            )
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0]

    @enforce_types
    def add_persona(self, persona: PersonaModel):
        with self.session_maker() as session:
            session.add(persona)
            session.commit()

    @enforce_types
    def get_human(self, name: str, user_id: uuid.UUID) -> Optional[HumanModel]:
        with self.session_maker() as session:
            results = (
                session.query(HumanModel)
                .filter(HumanModel.name == name)
                .filter(HumanModel.user_id == user_id)
                .all()
            )
            if len(results) == 0:
                return None
            assert len(results) == 1, f"Expected 1 result, got {len(results)}"
            return results[0]

    @enforce_types
    def add_human(self, human: HumanModel):
        with self.session_maker() as session:
            session.add(human)
            session.commit()

    @enforce_types
    def get_agent(
        self,
        agent_id: Optional[uuid.UUID] = None,
        agent_name: Optional[str] = None,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[AgentState]:
        with self.session_maker() as session:
            if agent_id:
                results = (
                    session.query(AgentModel).filter(AgentModel.id == agent_id).all()
                )
            else:
                assert (
                    agent_name is not None and user_id is not None
                ), "Must provide either agent_id or agent_name"
                results = (
                    session.query(AgentModel)
                    .filter(AgentModel.name == agent_name)
                    .filter(AgentModel.user_id == user_id)
                    .all()
                )

            if len(results) == 0:
                return None
            assert (
                len(results) == 1
            ), f"Expected 1 result, got {len(results)}"  # should only be one result
            return results[0].to_record()

    @enforce_types
    def list_agents(self, user_id: uuid.UUID) -> List[AgentState]:
        with self.session_maker() as session:
            results = (
                session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
            )
            return [r.to_record() for r in results]

    @enforce_types
    def update_agent(self, agent: AgentState):
        with self.session_maker() as session:
            session.query(AgentModel).filter(AgentModel.id == agent.id).update(
                vars(agent)
            )
            session.commit()

    @enforce_types
    def create_agent(self, agent: AgentState):
        # insert into agent table
        # make sure agent.name does not already exist for user user_id
        assert agent.state is not None, "Agent state must be provided"
        assert len(list(agent.state.keys())) > 0, "Agent state must not be empty"
        with self.session_maker() as session:
            if (
                session.query(AgentModel)
                .filter(AgentModel.name == agent.name)
                .filter(AgentModel.user_id == agent.user_id)
                .count()
                > 0
            ):
                raise ValueError(f"Agent with name {agent.name} already exists")
            session.add(AgentModel(**vars(agent)))
            session.commit()

    @enforce_types
    def get_api_key(self, api_key: str) -> Optional[Token]:
        with self.session_maker() as session:
            print("getting api key", api_key)
            print(
                [
                    r.token
                    for r in self.get_all_api_keys_for_user(user_id=uuid.UUID(int=0))
                ]
            )
            results = (
                session.query(TokenModel).filter(TokenModel.token == api_key).all()
            )
            print("results", [r.token for r in results])
            if len(results) == 0:
                return None
            assert (
                len(results) == 1
            ), f"Expected 1 result, got {len(results)}"  # should only be one result
            return results[0].to_record()

    @enforce_types
    def get_user_from_api_key(self, api_key: str) -> Optional[User]:
        """Get the user associated with a given API key"""
        token = self.get_api_key(api_key=api_key)
        print("got api key", token.token, token is None)
        if token is None:
            raise ValueError(f"Token {api_key} does not exist")
        else:
            print(
                isinstance(token.user_id, uuid.UUID),
                self.get_user(user_id=token.user_id),
            )
            return self.get_user(user_id=token.user_id)

    @enforce_types
    def get_all_api_keys_for_user(self, user_id: uuid.UUID) -> List[Token]:
        with self.session_maker() as session:
            results = (
                session.query(TokenModel).filter(TokenModel.user_id == user_id).all()
            )
            tokens = [r.to_record() for r in results]
            print([t.token for t in tokens])
            return tokens
