"""Compare SQLite reads with an in-process cache and a Redis cache.

Run with: uv run fastapi dev caching.py
"""

from contextlib import asynccontextmanager
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from sqlalchemy import Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DATABASE_URL = "sqlite:///./caching_demo.db"
REDIS_URL = "redis://localhost:6379/0"
CACHE_TTL_SECONDS = 60
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
local_cache: dict[int, dict[str, Any]] = {}


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    price_cents: Mapped[int] = mapped_column(Integer)


class ProductUpdate(BaseModel):
    name: str
    price_cents: int


def seed_database() -> None:
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        if session.scalar(select(Product.id).limit(1)) is None:
            session.add_all([Product(id=1, name="Mechanical keyboard", price_cents=7999), Product(id=2, name="Wireless mouse", price_cents=2499)])
            session.commit()


def read_product_from_sqlite(product_id: int) -> dict[str, Any]:
    # Deliberate demo delay; remove it in a real service.
    time.sleep(0.15)
    with Session(engine) as session:
        product = session.get(Product, product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        return {"id": product.id, "name": product.name, "price_cents": product.price_cents}


def timed(product: dict[str, Any], source: str, started: float) -> dict[str, Any]:
    return {**product, "source": source, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_database()
    app.state.redis = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        app.state.redis.ping()
    except Exception as error:
        app.state.redis.close()
        raise RuntimeError(f"Redis is unavailable at {REDIS_URL}") from error
    yield
    app.state.redis.close()


app = FastAPI(title="SQLite caching examples", lifespan=lifespan)


@app.get("/products/{product_id}/no-cache")
def no_cache(product_id: int) -> dict[str, Any]:
    """Example 1: SQLite is queried for every request."""
    started = time.perf_counter()
    return timed(read_product_from_sqlite(product_id), "sqlite", started)


@app.get("/products/{product_id}/local-cache")
def in_process_cache(product_id: int) -> dict[str, Any]:
    """Example 2: a dict cache, fast but isolated to one app process."""
    started = time.perf_counter()
    product = local_cache.get(product_id)
    if product is not None:
        return timed(product, "in_process_cache", started)
    product = read_product_from_sqlite(product_id)
    local_cache[product_id] = product
    return timed(product, "sqlite (local cache populated)", started)


@app.get("/products/{product_id}/redis-cache")
def redis_cache(product_id: int) -> dict[str, Any]:
    """Example 3: Redis cache, shared by multiple FastAPI processes."""
    started = time.perf_counter()
    key = f"product:{product_id}"
    redis: Redis = app.state.redis
    cached = redis.hgetall(key)
    if cached:
        return timed({"id": int(cached["id"]), "name": cached["name"], "price_cents": int(cached["price_cents"])}, "redis_cache", started)
    product = read_product_from_sqlite(product_id)
    redis.hset(key, mapping={name: str(value) for name, value in product.items()})
    redis.expire(key, CACHE_TTL_SECONDS)
    return timed(product, "sqlite (Redis cache populated)", started)


@app.put("/products/{product_id}")
def update_product(product_id: int, update: ProductUpdate) -> dict[str, str]:
    """Update SQLite and invalidate both cache variants."""
    with Session(engine) as session:
        product = session.get(Product, product_id)
        if product is None:
            raise HTTPException(status_code=404, detail="Product not found")
        product.name, product.price_cents = update.name, update.price_cents
        session.commit()
    local_cache.pop(product_id, None)
    app.state.redis.delete(f"product:{product_id}")
    return {"message": "Product updated; local and Redis cache entries removed."}


@app.post("/cache/clear")
def clear_caches() -> dict[str, str]:
    """Use before each Thunder Client timing comparison."""
    local_cache.clear()
    keys = app.state.redis.keys("product:*")
    deleted = app.state.redis.delete(*keys) if keys else 0
    return {"message": f"Cleared local cache and {deleted} Redis product cache entries."}
