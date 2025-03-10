import asyncio
import json
import uuid

from asyncio import AbstractEventLoop
from fastapi import APIRouter, Body, HTTPException, Depends
from functools import partial
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from server import SyncServer
from interface import QueuingInterface
from starlette.responses import StreamingResponse
from constants import JSON_ENSURE_ASCII
from auth_token import get_current_user


router = APIRouter()


class MessageRoleType(str, Enum):
    user = "user"
    system = "system"


class UserMessageResponse(BaseModel):
    messages: List[dict] = Field(
        ...,
        description="List of messages generated by the agent in response to the received message.",
    )


class UserMessageRequest(BaseModel):
    message: str = Field(
        ..., description="The message content to be processed by the agent."
    )
    role: MessageRoleType = Field(
        default=MessageRoleType.user,
        description="Role of the message sender (either 'user' or 'system')",
    )


def setup_agents_message_router(
    server: SyncServer, interface: QueuingInterface, password: str
):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.post(
        "/agents/{agent_id}/message",
        tags=["agents"],
        response_model=UserMessageResponse,
    )
    async def send_message(
        agent_id: uuid.UUID,
        request: UserMessageRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Process a user message and return the agent's response.

        This endpoint accepts a message from a user and processes it through the agent.
        It can optionally stream the response if 'stream' is set to True.
        """

        if request.role == "user" or request.role is None:
            message_func = server.user_message
        elif request.role == "system":
            message_func = server.system_message
        else:
            raise HTTPException(status_code=500, detail=f"Bad role {request.role}")

        try:
            # Start the generation process (similar to the non-streaming case)
            # This should be a non-blocking call or run in a background task
            # Check if server.user_message is an async function
            if asyncio.iscoroutinefunction(message_func):
                # Start the async task
                await asyncio.create_task(
                    message_func(
                        user_id=user_id,
                        agent_id=agent_id,
                        message=request.message,
                    )
                )
            else:

                def handle_exception(exception_loop: AbstractEventLoop, context):
                    error = context.get("exception") or context["message"]
                    print(f"handling asyncio exception {context}")
                    interface.error(str(error))

                # Run the synchronous function in a thread pool
                loop = asyncio.get_event_loop()
                loop.set_exception_handler(handle_exception)
                loop.run_in_executor(
                    None,
                    message_func,
                    user_id,
                    agent_id,
                    request.message,
                )

            async def formatted_message_generator():
                async for message in interface.message_generator():
                    formatted_message = f"data: {json.dumps(message, ensure_ascii=JSON_ENSURE_ASCII)}\n\n"
                    yield formatted_message
                    await asyncio.sleep(1)

            # Return the streaming response using the generator
            return StreamingResponse(
                formatted_message_generator(), media_type="text/event-stream"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
