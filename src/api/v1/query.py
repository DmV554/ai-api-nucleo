
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import the new factory function and data models
from src.core.strategies import get_strategy
from src.core.strategies import RAGInput

router = APIRouter()

# --- Pydantic Models for Request and Response ---

class QueryRequest(BaseModel):
    question: str
    collection_name: str
    strategy: str = "naive"  # Can be 'naive' or 'hybrid'
    top_k: int = 5

class DocumentResponse(BaseModel):
    content: str
    meta: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    documents: List[DocumentResponse]

# --- API Endpoint ---

@router.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Receives a user's question and processes it using the specified RAG strategy.
    
    - **collection_name**: The knowledge base to query against.
    - **strategy**: The RAG pipeline to use ('naive' or 'hybrid').
    - **top_k**: The number of documents to retrieve and/or rank.
    """
    try:
        # 1. Get the appropriate strategy instance from the factory
        strategy_instance = get_strategy(request.collection_name, request.strategy)
        
        # 2. Prepare the input for the strategy
        rag_input = RAGInput(question=request.question, top_k=request.top_k)

        # 3. Run the strategy in a separate thread to avoid blocking the event loop
        result = await asyncio.to_thread(strategy_instance.run, rag_input)

        # 4. Format the response
        response_docs = []
        for doc in result.documents:
            # The to_dict() method flattens the 'meta' dictionary.
            # We need to reconstruct it for the response model.
            content = doc.pop('content', '')
            # The rest of the items in the dictionary are the metadata
            meta = doc
            response_docs.append(DocumentResponse(content=content, meta=meta))

        return QueryResponse(answer=result.answer, documents=response_docs)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{request.collection_name}' not found. Please ensure it has been created.")
    except ValueError as e:
        # This will catch unsupported strategy names from our factory
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the full error for debugging
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")
