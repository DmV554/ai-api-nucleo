from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

from .pipeline_manager import pipeline_manager

# --- Pydantic Models for internal data transfer ---

class RAGInput(BaseModel):
    question: str
    top_k: int

class RAGResult(BaseModel):
    answer: str
    documents: List[Dict[str, Any]]

# --- Base Strategy Definition ---

class BaseRAGStrategy(ABC):
    """
    Abstract base class for a RAG strategy.
    It defines the interface for running a query and encapsulates the logic
    of building pipelines and processing results.
    """
    def __init__(self, collection_name: str):
        # Each strategy instance is tied to a specific collection
        # and will use the pipeline_manager to get its pipeline.
        self.pipeline = pipeline_manager.get_pipeline(collection_name, self.get_strategy_name())

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Returns the unique name of the strategy (e.g., 'naive', 'hybrid')."""
        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, request: RAGInput) -> Dict[str, Any]:
        """Prepares the input dictionary for the pipeline.run() method."""
        raise NotImplementedError

    @abstractmethod
    def extract_output(self, result: Dict[str, Any]) -> RAGResult:
        """Extracts and formats the final response from the pipeline output."""
        raise NotImplementedError
    
    @abstractmethod
    def _get_components_to_include(self) -> List[str]:
        """Specifies which pipeline components' outputs are needed."""
        raise NotImplementedError

    def run(self, request: RAGInput) -> RAGResult:
        """Executes the full RAG process for a given input."""
        pipeline_input = self.prepare_input(request)
        components_to_include = self._get_components_to_include()
        
        # The pipeline is run here
        result = self.pipeline.run(pipeline_input, include_outputs_from=components_to_include)
        
        return self.extract_output(result)

# --- Concrete Strategy Implementations ---

class NaiveRAGStrategy(BaseRAGStrategy):
    """Implements the 'naive' RAG strategy."""

    def get_strategy_name(self) -> str:
        return "naive"

    def prepare_input(self, request: RAGInput) -> Dict[str, Any]:
        return {
            "query_embedder": {"text": request.question},
            "retriever": {"top_k": request.top_k}
        }

    def _get_components_to_include(self) -> List[str]:
        return ["llm", "retriever"]

    def extract_output(self, result: Dict[str, Any]) -> RAGResult:
        answer = result["llm"]["replies"][0]
        docs = result.get("retriever", {}).get("documents", [])
        # Convert Haystack Document objects to dictionaries
        doc_dicts = [doc.to_dict() for doc in docs]
        return RAGResult(answer=answer, documents=doc_dicts)


class HybridRAGStrategy(BaseRAGStrategy):
    """Implements the 'hybrid' RAG strategy with a ranker."""

    def get_strategy_name(self) -> str:
        return "hybrid"

    def prepare_input(self, request: RAGInput) -> Dict[str, Any]:
        return {
            "dense_embedder": {"text": request.question},
            "sparse_embedder": {"text": request.question},
            "retriever": {"top_k": request.top_k},
            "ranker": {"query": request.question, "top_k": request.top_k}
        }
    
    def _get_components_to_include(self) -> List[str]:
        return ["llm", "ranker"]

    def extract_output(self, result: Dict[str, Any]) -> RAGResult:
        answer = result["llm"]["replies"][0]
        # In this pipeline, the ranker holds the final documents
        docs = result.get("ranker", {}).get("documents", [])
        # Convert Haystack Document objects to dictionaries
        doc_dicts = [doc.to_dict() for doc in docs]
        return RAGResult(answer=answer, documents=doc_dicts)


# --- Strategy Factory ---

_strategy_classes = {
    "naive": NaiveRAGStrategy,
    "hybrid": HybridRAGStrategy
}

def get_strategy(collection_name: str, strategy_name: str) -> BaseRAGStrategy:
    """
    Factory function to get an instance of a RAG strategy.

    Args:
        collection_name: The knowledge base to target.
        strategy_name: The name of the strategy to use.

    Returns:
        An instance of a BaseRAGStrategy subclass.
    
    Raises:
        ValueError: If the strategy_name is not supported.
    """
    strategy_class = _strategy_classes.get(strategy_name)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}. Supported strategies are: {list(_strategy_classes.keys())}")
    
    # Here we instantiate the strategy class for the given collection
    return strategy_class(collection_name)