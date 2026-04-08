from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.config import settings
from app.core.logger import log

# 1. Setup Base class untuk semua model SQLAlchemy
Base = declarative_base()

# 2. Setup Async Engine
# Format URL harus seperti: postgresql+asyncpg://user:password@host:port/dbname
engine = create_async_engine(
    settings.POSTGRES_URI,  # Sesuaikan dengan variabel config-mu
    echo=False,
    future=True,
)

# 3. Setup Async Session Maker
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# 4. Dependency untuk mendapatkan session database (sangat cocok digunakan di FastAPI Depends)
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# 5. Inisialisasi Database (membuat tabel)
async def init_db():
    try:
        async with engine.begin() as conn:
            # PENTING: Di production, lebih baik gunakan Alembic untuk migrasi database.
            # create_all ini biasanya hanya dipakai untuk development/testing awal.
            await conn.run_sync(Base.metadata.create_all)

        log.info("✅ PostgreSQL Database initialized via SQLAlchemy.")
    except Exception as e:
        log.error(f"❌ Failed to initialize database: {e}")
