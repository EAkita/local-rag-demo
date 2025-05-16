from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List, Dict, Any
from langchain_core.documents import Document

class MetadataFilteringFAISSRetriever(VectorStoreRetriever):
    def __init__(self, vectorstore: FAISS, filter: Dict[str, Any] = None, k: int = 5):
        self.vectorstore = vectorstore
        self.k = k
        self.filter = filter or {}

    def _apply_filter(self, docs: List[Document]) -> List[Document]:
        filtered = []
        for doc in docs:
            metadata = doc.metadata
            keep = True
            for key, val in self.filter.items():
                if isinstance(val, dict) and "$gte" in val:
                    if metadata.get(key, "") < val["$gte"]:
                        keep = False
                        break
                elif metadata.get(key) != val:
                    keep = False
                    break
            if keep:
                filtered.append(doc)
        return filtered

    def invoke(self, query: str) -> List[Document]:
        all_docs = self.vectorstore.docstore._dict.values()
        filtered_docs = self._apply_filter(all_docs)

        if not filtered_docs:
            return []

        temp_store = FAISS.from_documents(filtered_docs, self.vectorstore.embedding_function)
        return temp_store.similarity_search(query, k=self.k)
