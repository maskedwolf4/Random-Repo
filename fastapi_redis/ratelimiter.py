"""A Redis fixed-window rate limiter. Run: uv run fastapi dev ratelimiter.py --port 8001"""

from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException, Request, Response
from redis import Redis

REDIS_URL = "redis://localhost:6379/0"
LIMIT, WINDOW_SECONDS = 5, 60


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


app = FastAPI(title="Redis rate limiter", lifespan=lifespan)


def enforce_limit(request: Request, response: Response) -> None:
    client_ip = request.client.host if request.client else "unknown"
    window = int(time.time() // WINDOW_SECONDS)
    redis: Redis = app.state.redis
    key = f"rate-limit:{client_ip}:{window}"
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, WINDOW_SECONDS)
    ttl = max(redis.ttl(key), 0)
    response.headers["X-RateLimit-Limit"] = str(LIMIT)
    response.headers["X-RateLimit-Remaining"] = str(max(LIMIT - count, 0))
    response.headers["X-RateLimit-Reset"] = str(ttl)
    if count > LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests", headers={"Retry-After": str(ttl)})


@app.get("/limited")
def limited_endpoint(request: Request, response: Response) -> dict[str, str]:
    enforce_limit(request, response)
    return {"message": "Request accepted. You have 5 requests per 60-second window."}


@app.post("/rate-limit/reset")
def reset_for_demo(request: Request) -> dict[str, str]:
    """Local demo convenience endpoint; do not expose in production."""
    client_ip = request.client.host if request.client else "unknown"
    keys = app.state.redis.keys(f"rate-limit:{client_ip}:*")
    removed = app.state.redis.delete(*keys) if keys else 0
    return {"message": f"Removed {removed} rate-limit key(s) for {client_ip}."}
