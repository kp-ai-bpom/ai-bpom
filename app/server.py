from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.llm import init_llm
from app.core.logger import log


def _patch_openapi_for_swagger(schema: dict) -> dict:
    """Patch the OpenAPI schema so Swagger UI shows file-upload pickers.

    FastAPI 0.100+ generates OpenAPI 3.1 schemas that use
    ``contentMediaType: application/octet-stream`` for binary fields.
    Swagger UI (even 5.x) does NOT reliably render file pickers for this
    notation — it requires the OpenAPI 3.0 ``format: binary`` convention.

    This function:
    1. Downgrades the top-level ``openapi`` key from ``3.1.0`` to ``3.0.3``
       so Swagger UI uses its 3.0 renderer (which always shows file pickers).
    2. Replaces every ``type: string, contentMediaType: application/octet-stream``
       with ``type: string, format: binary`` and removes ``contentMediaType``.
    3. Removes ``$schema`` keys (not valid in OpenAPI 3.0).
    4. Converts ``prefixItems`` (3.1 tuple validation) to plain ``items`` style.
    """
    # 1. Downgrade OpenAPI version
    if schema.get("openapi", "").startswith("3.1"):
        schema["openapi"] = "3.0.3"

    # 2. Remove JSON Schema keys not valid in OpenAPI 3.0
    schema.pop("$schema", None)

    # 3. Walk components and paths to patch binary markers
    components = schema.get("components", {}).get("schemas", {})
    for _name, defn in components.items():
        _patch_schema_node(defn)

    for _path, methods in schema.get("paths", {}).items():
        for _method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            body = operation.get("requestBody", {})
            for _media, media_def in body.get("content", {}).items():
                inline_schema = media_def.get("schema", {})
                if inline_schema.get("$ref"):
                    continue
                _patch_schema_node(inline_schema)

    return schema


def _patch_schema_node(node: dict) -> None:
    """Recursively patch a schema node for Swagger UI compatibility."""
    if not isinstance(node, dict):
        return

    # Remove OpenAPI 3.1-only keywords
    node.pop("$schema", None)

    # Convert contentMediaType binary → format: binary
    if (
        node.get("type") == "string"
        and node.get("contentMediaType") == "application/octet-stream"
    ):
        node.pop("contentMediaType", None)
        node["format"] = "binary"

    # Convert prefixItems (3.1) → items (3.0) if present
    if "prefixItems" in node and "items" not in node:
        node["items"] = node.pop("prefixItems")[0] if node["prefixItems"] else {}
    elif "prefixItems" in node:
        node.pop("prefixItems", None)

    # Recurse into items
    items = node.get("items")
    if isinstance(items, dict):
        _patch_schema_node(items)

    # Recurse into properties
    for _key, val in node.get("properties", {}).items():
        if isinstance(val, dict):
            _patch_schema_node(val)

    # Recurse into anyOf / allOf / oneOf
    for combiner in ("anyOf", "allOf", "oneOf"):
        for sub in node.get(combiner, []):
            if isinstance(sub, dict):
                _patch_schema_node(sub)


def create_app() -> FastAPI:
    """
    Application Factory: Merakit dan mengembalikan instance FastAPI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan Manager untuk startup dan shutdown events."""

        log.info("Starting up server...")

        # Initialize LLM
        init_llm()

        yield

        log.info("Shutting down server...")

    # Inisialisasi instance FastAPI
    app = FastAPI(
        title="AI Service API BPOM",
        description="AI Service API for Chatbot, Pemetaan Suksesor, and Penilaian Suksesor",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_api_route(
        "/", lambda: {"message": "Welcome to AI Services API!"}, methods=["GET"]
    )
    app.include_router(api_router, prefix="/api")

    # Override openapi() to patch the schema for Swagger UI file-upload support.
    # OpenAPI 3.1's `contentMediaType` notation is not reliably rendered by
    # Swagger UI, so we downgrade to 3.0.3 and use `format: binary`.
    _original_openapi = app.openapi

    def _custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = _original_openapi()
        schema = _patch_openapi_for_swagger(schema)
        app.openapi_schema = schema
        return schema

    app.openapi = _custom_openapi

    return app