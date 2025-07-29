
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
import os

from src.config import settings

def get_document_store(collection_name: str, use_sparse: bool = False) -> QdrantDocumentStore:
    """
    Initializes and returns a QdrantDocumentStore for a specific collection.

    This function centralizes the connection to Qdrant, ensuring consistent
    configuration across the application.

    Args:
        collection_name: The name of the Qdrant collection to connect to.
        use_sparse: Whether to enable sparse embeddings for hybrid search.

    Returns:
        An instance of QdrantDocumentStore connected to the specified collection.
    """
    # The path where the local Qdrant database for this specific collection will be stored.
    db_path = os.path.join(settings.VECTOR_STORE_PATH, collection_name)

    # --- FIX: Check if the knowledge base directory actually exists ---
    # If not, raise an error instead of letting Qdrant create an empty one.
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Knowledge base '{collection_name}' not found at path: {db_path}. "
            f"Please run the ingestion script first for this collection."
        )

    return QdrantDocumentStore(
        path=db_path,
        index=collection_name,
        use_sparse_embeddings=use_sparse,
        embedding_dim=settings.EMBEDDING_DIM,
        # recreate_index should be handled by the ingestion script, not the service.
        # sparse_idf is also typically calculated during ingestion.
    )
