"""Cookie sessions whose data is stored in Redis. Run: uv run fastapi dev sessions.py --port 8002"""

from contextlib import asynccontextmanager
import json
import secrets

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel
from redis import Redis

REDIS_URL = "redis://localhost:6379/0"
SESSION_TTL_SECONDS, COOKIE_NAME = 30 * 60, "session_id"


class Login(BaseModel):
    username: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        app.state.redis.ping()
    except Exception as error:
        app.state.redis.close()
        raise RuntimeError(f"Redis is unavailable at {REDIS_URL}") from error
    yield
    app.state.redis.close()


app = FastAPI(title="Redis sessions", lifespan=lifespan)


def session_for(request: Request) -> tuple[str, dict[str, str]]:
    session_id = request.cookies.get(COOKIE_NAME)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Please log in first")
    payload = app.state.redis.get(f"session:{session_id}")
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid")
    app.state.redis.expire(f"session:{session_id}", SESSION_TTL_SECONDS)
    return session_id, json.loads(payload)


@app.post("/login")
def login(body: Login, response: Response) -> dict[str, str | int]:
    """Demo login: accepts any username; real apps must verify credentials."""
    session_id = secrets.token_urlsafe(32)
    app.state.redis.setex(f"session:{session_id}", SESSION_TTL_SECONDS, json.dumps({"username": body.username}))
    response.set_cookie(key=COOKIE_NAME, value=session_id, max_age=SESSION_TTL_SECONDS, httponly=True, samesite="lax", secure=False)
    return {"message": f"Logged in as {body.username}", "expires_in_seconds": SESSION_TTL_SECONDS}


@app.get("/me")
def me(request: Request) -> dict[str, str]:
    _, session = session_for(request)
    return {"username": session["username"], "message": "Loaded from the Redis-backed session."}


@app.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    session_id, _ = session_for(request)
    app.state.redis.delete(f"session:{session_id}")
    response.delete_cookie(COOKIE_NAME)
    return {"message": "Logged out and Redis session deleted."}
