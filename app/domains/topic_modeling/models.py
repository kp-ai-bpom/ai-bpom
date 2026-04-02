from beanie import Document
from pydantic import Field


class DocumentsModel(Document):
    """Model untuk menyimpan hasil analisis topik dari dokumen (misalnya tweet)"""

    projectId: str = Field(
        ...,
        description="Project ID for data persistence",
        json_schema_extra={"example": "proj_123456"},
    )
    full_text: str = Field(
        ...,
        description="Full text of the document",
        json_schema_extra={
            "example": "logo byond bsi muncul banyak delapan kali lanjut"
        },
    )
    raw_text: str = Field(
        ...,
        description="Raw text of the document",
        json_schema_extra={
            "example": "@bankbsi_id 8 kali muncul Logo BYOND by BSI #PahamJadiBerkah Gasskeuunnn...!!! @Komaria__ria @zzahraxyz @nurshofia1010"
        },
    )
    username: str = Field(
        ...,
        description="Username of the document",
        json_schema_extra={"example": "brugmansia_"},
    )
    tweet_url: str = Field(
        ...,
        description="Tweet URL of the document",
        json_schema_extra={
            "example": "https://x.com/brugmansia_/status/1888336969257889924"
        },
    )
    topic: int = Field(..., description="Topic ID")
    probability: float = Field(..., description="Probability of the topic")

    class Settings:
        name = "documents"  # Nama collection di MongoDB
        indexes = [
            "projectId",  # Index untuk mempercepat filter pencarian berdasarkan projectId
            "topic",  # Index untuk mempercepat filter pencarian berdasarkan topic
        ]


class TopicsModel(Document):
    """Model untuk menyimpan informasi topik"""

    topicId: int = Field(
        ...,
        description="Topic ID for data persistence",
        json_schema_extra={"example": 1},
    )
    projectId: str = Field(
        ...,
        description="Project ID for data persistence",
        json_schema_extra={"example": "proj_123456"},
    )
    context: str = Field(
        ...,
        description="Context of the topic",
        json_schema_extra={"example": "Logo BYOND by BSI"},
    )
    words: list = Field(
        ...,
        description="Words of the topic",
        json_schema_extra={"example": ["Logo", "BYOND", "BSI"]},
    )
    keyword: str = Field(
        ...,
        description="Keyword of the topic",
        json_schema_extra={"example": "byond bsi"},
    )

    class Settings:
        name = "topics"  # Nama collection di MongoDB
        indexes = [
            "projectId",  # Index untuk mempercepat filter pencarian berdasarkan projectId
        ]
