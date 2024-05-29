import uuid

from constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from sqlmodel import Field, SQLModel
from typing import Optional, List, Dict
from utils import get_human_text, get_persona_text
from enum import Enum
from datetime import datetime
from sqlalchemy import JSON, Column
from utils import get_utc_time
from sqlalchemy_utils import ChoiceType
from pydantic import BaseModel, ConfigDict


class LLMConfigModel(BaseModel):
    model: Optional[str] = "gpt-4"
    model_endpoint: Optional[str] = "https://api.openai.com/v1"
    model_wrapper: Optional[str] = None
    context_window: Optional[int] = None

    # FIXME hack to silence pydantic protected namespace warning
    model_config = ConfigDict(protected_namespaces=())


class HumanModel(SQLModel, table=True):
    text: str = Field(
        default=get_human_text(DEFAULT_HUMAN), description="The human text."
    )
    name: str = Field(..., description="The name of the human.")
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="The unique identifier of the human.",
        primary_key=True,
    )
    user_id: Optional[uuid.UUID] = Field(
        ..., description="The unique identifier of the user associated with the human."
    )


class PersonaModel(SQLModel, table=True):
    text: str = Field(
        default=get_persona_text(DEFAULT_PERSONA), description="The persona text."
    )
    name: str = Field(..., description="The name of the persona.")
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="The unique identifier of the persona.",
        primary_key=True,
    )
    user_id: Optional[uuid.UUID] = Field(
        ...,
        description="The unique identifier of the user associated with the persona.",
    )


class ToolModel(SQLModel, table=True):
    name: str = Field(..., description="The name of the function.")
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="The unique identifier of the function.",
        primary_key=True,
    )
    tags: List[str] = Field(sa_column=Column(JSON), description="Metadata tags.")
    source_type: Optional[str] = Field(None, description="The type of the source code.")
    source_code: Optional[str] = Field(
        ..., description="The source code of the function."
    )

    json_schema: Dict = Field(
        default_factory=dict,
        sa_column=Column(JSON),
        description="The JSON schema of the function.",
    )

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True


class JobStatus(str, Enum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobModel(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="The unique identifier of the job.",
        primary_key=True,
    )
    # status: str = Field(default="created", description="The status of the job.")
    status: JobStatus = Field(
        default=JobStatus.created,
        description="The status of the job.",
        sa_column=Column(ChoiceType(JobStatus)),
    )
    created_at: datetime = Field(
        default_factory=get_utc_time,
        description="The unix timestamp of when the job was created.",
    )
    completed_at: Optional[datetime] = Field(
        None, description="The unix timestamp of when the job was completed."
    )
    user_id: uuid.UUID = Field(
        ..., description="The unique identifier of the user associated with the job."
    )
    metadata_: Optional[dict] = Field(
        {}, sa_column=Column(JSON), description="The metadata of the job."
    )
