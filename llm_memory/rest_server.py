import typer
import secrets

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from message import setup_agents_message_router
from server import SyncServer
from interface import QueuingInterface

interface: QueuingInterface = QueuingInterface()
server: SyncServer = SyncServer(default_interface=interface)


API_PREFIX = "/memgpt_api"

password = secrets.token_urlsafe(16)
typer.secho(
    f"Generated admin server password for this session: {password}",
    fg=typer.colors.GREEN,
)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    setup_agents_message_router(server, interface, password), prefix=API_PREFIX
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="0.0.0.0", port=8080)
