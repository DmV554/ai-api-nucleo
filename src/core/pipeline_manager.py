from typing import Dict
from haystack import Pipeline

# Import the pipeline building functions
from .pipelines import build_naive_rag_pipeline, build_hybrid_rag_pipeline

class PipelineManager:
    """
    Manages the lifecycle of Haystack pipelines.

    This class ensures that pipelines are built only once and are cached in memory
    for reuse across multiple API requests. This is critical for performance as
    it avoids reloading heavy models on every call.
    """
    _pipelines: Dict[str, Pipeline] = {}

    @classmethod
    def get_pipeline(cls, collection_name: str, strategy: str) -> Pipeline:
        """
        Retrieves a pipeline from the cache. If not found, it builds, caches,
        and then returns the pipeline.

        Args:
            collection_name: The name of the knowledge base (collection) to target.
            strategy: The RAG strategy to use ('naive' or 'hybrid').

        Returns:
            A ready-to-use Haystack Pipeline instance.
        """
        cache_key = f"{collection_name}_{strategy}"
        
        if cache_key not in cls._pipelines:
            print(f"INFO: Pipeline for '{cache_key}' not found in cache. Building...")
            
            if strategy == "naive":
                pipeline = build_naive_rag_pipeline(collection_name)
            elif strategy == "hybrid":
                pipeline = build_hybrid_rag_pipeline(collection_name)
            else:
                raise ValueError(f"Unsupported RAG strategy: '{strategy}' ")
            
            cls._pipelines[cache_key] = pipeline
            print(f"INFO: Pipeline for '{cache_key}' built and cached successfully.")
        
        return cls._pipelines[cache_key]

# A single, globally accessible instance of the manager.
# The API will interact with this instance.
pipeline_manager = PipelineManager()