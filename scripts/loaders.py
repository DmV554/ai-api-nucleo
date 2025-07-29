from abc import ABC, abstractmethod
from typing import List
import os
import json
import pandas as pd
from haystack.dataclasses import Document

class BaseLoader(ABC):
    """Clase base abstracta para los cargadores de documentos."""
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """Carga documentos desde una ruta de origen y los devuelve como objetos Document de Haystack."""
        raise NotImplementedError

class ScraperJSONLLoader(BaseLoader):
    """Carga documentos desde la salida del scraper en formato .jsonl."""
    def load(self, path: str) -> List[Document]:
        jsonl_path = os.path.join(path, "dataset.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"Error: No se encontró 'dataset.jsonl' en {path}")
            return []

        docs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    metadata = {
                        "url": item.get("url", ""), "title": item.get("title", ""),
                        "timestamp": item.get("timestamp", ""), "content_hash": item.get("content_hash", "")
                    }
                    docs.append(Document(content=item.get("content", ""), meta=metadata))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Omitiendo línea en .jsonl por error: {e}")
        return docs

class CsvQALoader(BaseLoader):
    """Carga documentos desde un archivo CSV de tipo Pregunta/Respuesta."""
    def load(self, path: str) -> List[Document]:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo CSV en {path}")
            return []

        docs = []
        required_cols = ['answer', 'question']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: El CSV debe tener las columnas 'answer' y 'question'.")
            return []

        for _, row in df.iterrows():
            content = row.get('answer', '')
            if isinstance(content, str) and content.strip():
                metadata = {
                    'question': row.get('question', ''), 'title': row.get('title', ''),
                    'url': row.get('url', ''), 'source': row.get('source', '')
                }
                docs.append(Document(content=content, meta=metadata))
        return docs

# --- Fábrica de Cargadores ---

def get_loader(path: str) -> BaseLoader:
    """
    Función de fábrica que devuelve la instancia del cargador apropiado según la ruta.
    """
    if os.path.isdir(path):
        print(f"Detectado directorio: '{path}'. Usando cargador para formato scraper (jsonl).")
        return ScraperJSONLLoader()
    elif path.lower().endswith('.csv'):
        print(f"Detectado archivo CSV: '{path}'. Usando cargador para formato Q&A.")
        return CsvQALoader()
    # Para añadir un nuevo cargador (ej. PDF), añade aquí la condición:
    # elif path.lower().endswith('.pdf'):
    #     return PDFLoader()
    else:
        raise ValueError(f"Ruta no soportada o formato de archivo desconocido: {path}")
