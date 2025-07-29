
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file before other imports
load_dotenv()

# Importa la configuración centralizada
from src.config import settings

# Importaremos los routers de la API aquí cuando estén creados.
# Por ahora, esta línea está comentada para evitar errores.
from src.api.v1.query import router as query_router

# Crea la instancia de la aplicación FastAPI
app = FastAPI(
    title="AI API Modules",
    description="API modular para procesos de Retrieval-Augmented Generation (RAG).",
    version="0.1.0"
)

# Aquí incluiremos los routers de la API.
# Se descomentará cuando el router de consulta sea creado.
app.include_router(query_router, prefix=settings.API_V1_STR, tags=["Query"])

@app.get("/", tags=["Status"])
async def read_root():
    """
    Endpoint raíz para verificar que la API está funcionando.
    """
    return {"message": "Servicio AI API Modules está activo."}

# El método preferido para ejecutar la app es con: uvicorn main:app --reload
# Este bloque se mantiene para permitir la ejecución directa del script (python main.py)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
