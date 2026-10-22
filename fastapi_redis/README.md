# FastAPI + Redis examples

Three independent FastAPI apps show SQLite caching, Redis rate limiting, and Redis-backed sessions. Redis must already be running through your Docker Compose setup; the examples use `redis://localhost:6379/6379`.

## Start the apps

```bash
uv sync
uv run fastapi dev caching.py
```

Open `http://127.0.0.1:8000/docs`. The first start creates `caching_demo.db` and seeds two products.

Run the other examples in separate terminals:

```bash
uv run fastapi dev ratelimiter.py --port 8001
uv run fastapi dev sessions.py --port 8002
```

If your Docker Compose Redis host or port differs, change `REDIS_URL` in each file.

## 1. Cache timing in Thunder Client

Send `GET` requests for product `1` to the caching app:

| Example | URL | Expected result |
| --- | --- | --- |
| No cache | `http://127.0.0.1:8000/products/1/no-cache` | Every request is about 150 ms; `source` is `sqlite`. |
| In-process cache | `http://127.0.0.1:8000/products/1/local-cache` | First request is about 150 ms; later requests are fast with `source: in_process_cache`. |
| Redis cache | `http://127.0.0.1:8000/products/1/redis-cache` | First request is about 150 ms; later requests are fast with `source: redis_cache`. |

Before each comparison, send `POST http://127.0.0.1:8000/cache/clear`. Compare repeated calls using Thunder Client's response-time display. The 150 ms delay is deliberately included to make the local SQLite difference visible; remove it in real applications.

To demonstrate invalidation, update a product and then call either cache endpoint again:

```http
PUT http://127.0.0.1:8000/products/1
Content-Type: application/json

{"name":"Mechanical keyboard v2","price_cents":8999}
```

## 2. Rate limiter

Send `GET http://127.0.0.1:8001/limited` six times in one minute. Calls 1–5 return `200`; call 6 returns `429 Too Many Requests`. Inspect `X-RateLimit-Remaining`, `X-RateLimit-Reset`, and `Retry-After` headers. Use `POST http://127.0.0.1:8001/rate-limit/reset` for a fresh local-demo window.

The limiter uses Redis `INCR` and `EXPIRE`, keyed by client IP and fixed 60-second window, so multiple FastAPI workers share one count.

## 3. Redis sessions

In Thunder Client, enable its cookie jar, then:

1. `POST http://127.0.0.1:8002/login` with JSON body `{"username":"meet"}`.
2. `GET http://127.0.0.1:8002/me`; Thunder Client sends the `session_id` cookie automatically.
3. `POST http://127.0.0.1:8002/logout`.
4. Call `GET /me` again; it returns `401` because the Redis record and cookie are gone.

Only an opaque session ID is in the HTTP-only cookie. User data stays in Redis for 30 minutes and its expiry refreshes on `/me`. Login is intentionally a demo, not real authentication: production code must validate credentials, use HTTPS (`secure=True`), and apply CSRF protection where appropriate.
