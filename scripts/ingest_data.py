

import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file before anything else.
# This makes them available to the entire application.
load_dotenv()

from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder

# Import the new factory for document embedders
from src.core.pipelines import get_document_embedder
from src.config import settings
from .loaders import get_loader

# --- Script Principal de Ingesta ---

def run_ingestion_pipeline(collection_name: str, data_path: str, use_sparse: bool, policy: str):
    print("--- Iniciando Pipeline de Ingesta de Datos ---")
    
    # 1. Cargar documentos usando la fábrica de cargadores
    try:
        loader = get_loader(data_path)
        raw_documents = loader.load(data_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not raw_documents:
        print("Error: No se encontraron documentos para procesar. Abortando.")
        return
    print(f"Se encontraron {len(raw_documents)} documentos.")

    # 2. Configurar el DocumentStore
    db_path = os.path.join(settings.VECTOR_STORE_PATH, collection_name)
    document_store = QdrantDocumentStore(
        path=db_path, index=collection_name, 
        # The embedding dimension is now conditional
        embedding_dim=1536 if settings.EMBEDDING_PROVIDER == 'openai' else settings.EMBEDDING_DIM,
        use_sparse_embeddings=use_sparse, sparse_idf=True
    )
    write_policy = DuplicatePolicy[policy.upper()]

    # 3. Construir y ejecutar el pipeline de ingesta
    ingestion_pipeline = Pipeline()
    ingestion_pipeline.add_component("cleaner", DocumentCleaner())
    ingestion_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200, split_overlap=20))
    
    # Get the dense embedder from our factory
    dense_embedder = get_document_embedder()

    if use_sparse:
        print("Modo Híbrido: Construyendo pipeline de ingesta híbrida.")
        sparse_embedder = FastembedSparseDocumentEmbedder(model=settings.SPARSE_EMBEDDING_MODEL)
        ingestion_pipeline.add_component("sparse_embedder", sparse_embedder)
        ingestion_pipeline.add_component("dense_embedder", dense_embedder)
        ingestion_pipeline.add_component("writer", DocumentWriter(document_store, policy=write_policy))

        ingestion_pipeline.connect("cleaner.documents", "splitter.documents")
        ingestion_pipeline.connect("splitter.documents", "sparse_embedder.documents")
        ingestion_pipeline.connect("sparse_embedder.documents", "dense_embedder.documents")
        ingestion_pipeline.connect("dense_embedder.documents", "writer.documents")
    else:
        print("Modo Denso: Construyendo pipeline de ingesta densa.")
        ingestion_pipeline.add_component("dense_embedder", dense_embedder)
        ingestion_pipeline.add_component("writer", DocumentWriter(document_store, policy=write_policy))

        ingestion_pipeline.connect("cleaner.documents", "splitter.documents")
        ingestion_pipeline.connect("splitter.documents", "dense_embedder.documents")
        ingestion_pipeline.connect("dense_embedder.documents", "writer.documents")

    print(f"Ejecutando pre-procesamiento e indexación con política: '{policy.upper()}'...")
    ingestion_pipeline.run({"cleaner": {"documents": raw_documents}})

    print(f"\n✅ Ingesta completada. La base de conocimiento '{collection_name}' está lista.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crea una KB vectorial desde la salida del scraper o un CSV.")
    parser.add_argument("collection_name", type=str, help="Nombre de la colección (KB).")
    parser.add_argument("data_path", type=str, help="Ruta al directorio del scraper o al archivo .csv.")
    parser.add_argument("--hybrid", action="store_true", help="Genera embeddings dispersos y densos para búsqueda híbrida.")
    parser.add_argument(
        "--policy", 
        type=str, 
        default="overwrite", 
        choices=["overwrite", "skip"],
        help="Política de escritura: 'overwrite' para reemplazar documentos existentes, 'skip' para ignorarlos."
    )

    args = parser.parse_args()
    run_ingestion_pipeline(args.collection_name, args.data_path, args.hybrid, args.policy)




