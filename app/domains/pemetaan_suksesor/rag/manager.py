from strands import tool

from app.core.logger import log

from .graph.main import GraphRAG
from .hybrid.main import HybridRAG
from .vector.main import VectorRAG


class RAGManager:
    """Singleton manager for RAG instances and tool functions."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        self._vector_rag = VectorRAG()
        self._graph_rag = GraphRAG()
        self._hybrid_rag = HybridRAG(self._vector_rag, self._graph_rag)
        self._tools = self._create_tools()
        self._initialized = True
        log.info("🧠 RAG Manager initialized with Vector, Graph, and Hybrid RAG")

    @property
    def vector_rag(self) -> VectorRAG:
        self.initialize()
        return self._vector_rag

    @property
    def graph_rag(self) -> GraphRAG:
        self.initialize()
        return self._graph_rag

    @property
    def hybrid_rag(self) -> HybridRAG:
        self.initialize()
        return self._hybrid_rag

    @property
    def tools(self) -> list:
        self.initialize()
        return self._tools

    def _create_tools(self) -> list:
        vector_rag = self._vector_rag
        graph_rag = self._graph_rag
        hybrid_rag = self._hybrid_rag

        @tool
        def search_vector_rag(query: str) -> str:
            """Cari konteks semantik tentang jabatan, tugas, fungsi, persyaratan.
            Gunakan untuk pertanyaan deskriptif."""
            try:
                return vector_rag.retrieve(query)
            except Exception as e:
                return f"Error VectorRAG: {e}"

        @tool
        def search_graph_rag(entity: str) -> str:
            """Cari relasi antar jabatan/unit/fungsi.
            Gunakan untuk pertanyaan tentang atasan, bawahan, hubungan."""
            try:
                return graph_rag.retrieve(entity)
            except Exception as e:
                return f"Error GraphRAG: {e}"

        @tool
        def search_hybrid_rag(query: str, entity: str) -> str:
            """Cari konteks lengkap (semantik + relasi).
            Gunakan untuk pertanyaan kompleks yang butuh keduanya."""
            try:
                return hybrid_rag.retrieve(query, entity)
            except Exception as e:
                return f"Error HybridRAG: {e}"

        return [search_vector_rag, search_graph_rag, search_hybrid_rag]


def get_rag_manager() -> RAGManager:
    return RAGManager()