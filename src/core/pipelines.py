
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever, QdrantHybridRetriever
from pathlib import Path

# Import OpenAI components
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder

# Import Haystack's secret management
from haystack.utils import Secret

from src.config import settings
from src.services.document_store import get_document_store

# --- Component Factories ---

def get_llm():
    """Factory to get the appropriate LLM based on settings."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        print(f"Using LLM provider: OpenAI (model: {settings.OPENAI_LLM_MODEL})")
        # Use Secret.from_env_var to securely pass the API key
        return OpenAIGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), model=settings.OPENAI_LLM_MODEL)
    elif provider == "ollama":
        print(f"Using LLM provider: Ollama (model: {settings.OLLAMA_LLM_MODEL})")
        return OllamaGenerator(model=settings.OLLAMA_LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

def get_text_embedder():
    """Factory to get the appropriate text embedder for queries."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openai":
        print(f"Using Text Embedder provider: OpenAI (model: {settings.OPENAI_EMBEDDING_MODEL})")
        return OpenAITextEmbedder(api_key=Secret.from_env_var("OPENAI_API_KEY"), model=settings.OPENAI_EMBEDDING_MODEL)
    elif provider == "local":
        print(f"Using Text Embedder provider: Local (model: {settings.LOCAL_EMBEDDING_MODEL})")
        return SentenceTransformersTextEmbedder(model=settings.LOCAL_EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported Embedding provider: {settings.EMBEDDING_PROVIDER}")

def get_document_embedder():
    """Factory to get the appropriate document embedder for ingestion."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openai":
        print(f"Using Document Embedder provider: OpenAI (model: {settings.OPENAI_EMBEDDING_MODEL})")
        return OpenAIDocumentEmbedder(api_key=Secret.from_env_var("OPENAI_API_KEY"), model=settings.OPENAI_EMBEDDING_MODEL)
    elif provider == "local":
        print(f"Using Document Embedder provider: Local (model: {settings.LOCAL_EMBEDDING_MODEL})")
        return SentenceTransformersDocumentEmbedder(model=settings.LOCAL_EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported Embedding provider: {settings.EMBEDDING_PROVIDER}")

# --- Prompt Template Loading (no changes) ---

def load_prompt_template(template_name: str = "expert_rag_template.txt") -> str:
    """Loads a prompt template from the 'prompts' directory."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    prompt_path = base_dir / "prompts" / template_name
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt template not found at: {prompt_path}")

prompt_template = load_prompt_template()

# --- Pipeline Definitions ---

def build_naive_rag_pipeline(collection_name: str) -> Pipeline:
    """
    Builds and returns a simple RAG pipeline using a dense vector retriever.
    It now uses the component factories to select the LLM and Embedder.
    """
    document_store = get_document_store(collection_name, use_sparse=False)
    
    text_embedder = get_text_embedder()
    retriever = QdrantEmbeddingRetriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = get_llm()

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("query_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    return rag_pipeline

def build_hybrid_rag_pipeline(collection_name: str) -> Pipeline:
    """
    Builds and returns a hybrid RAG pipeline with a re-ranker.
    It now uses the component factories.
    """
    document_store = get_document_store(collection_name, use_sparse=True)

    sparse_embedder = FastembedSparseTextEmbedder(model=settings.SPARSE_EMBEDDING_MODEL)
    dense_embedder = get_text_embedder()
    retriever = QdrantHybridRetriever(document_store=document_store)
    ranker = SentenceTransformersSimilarityRanker(model=settings.RANKER_MODEL)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = get_llm()

    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component("sparse_embedder", sparse_embedder)
    hybrid_pipeline.add_component("dense_embedder", dense_embedder)
    hybrid_pipeline.add_component("retriever", retriever)
    hybrid_pipeline.add_component("ranker", ranker)
    hybrid_pipeline.add_component("prompt_builder", prompt_builder)
    hybrid_pipeline.add_component("llm", llm)

    hybrid_pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
    hybrid_pipeline.connect("dense_embedder.embedding", "retriever.query_embedding")
    hybrid_pipeline.connect("retriever.documents", "ranker.documents")
    hybrid_pipeline.connect("ranker.documents", "prompt_builder.documents")
    hybrid_pipeline.connect("prompt_builder", "llm")

    return hybrid_pipeline
