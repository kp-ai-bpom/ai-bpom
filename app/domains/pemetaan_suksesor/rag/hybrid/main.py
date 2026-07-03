class HybridRAG:
    def __init__(self, vector_rag, graph_rag):
        self.vector_rag = vector_rag
        self.graph_rag = graph_rag

    def retrieve(self, query: str, entity: str) -> str:
        vector_context = self.vector_rag.retrieve(query)
        graph_context = self.graph_rag.retrieve(entity)
        return (
            f"[Konteks VectorRAG]\n{vector_context}\n\n"
            f"[Konteks GraphRAG]\n{graph_context}"
        )