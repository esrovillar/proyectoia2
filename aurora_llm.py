"""
aurora_llm.py: la configuración y el modelo de LangChain, en un solo lugar.
USE_LOCAL_LLM en .env decide si le hablas a LM Studio o a OpenAI.
"""

# os lee las variables de entorno; Any sirve para tipar el diccionario de configuración
import os
from typing import Any

# openai: el SDK, para preguntarle a LM Studio qué modelos tiene y para sus excepciones
import openai
# load_dotenv carga el archivo .env; ChatOpenAI es el modelo como pieza de LangChain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def _env_text(name: str, default: str) -> str:
    """La variable como texto; si no existe o está vacía (MODEL=), el valor por defecto."""
    return os.getenv(name, "").strip() or default


def _env_number(name: str, default: str, cast: type, minimum: float) -> Any:
    """La variable convertida a número (int o float). Lanza ValueError que dice cuál está mal."""
    raw = _env_text(name, default)
    try:
        value = cast(raw)
    except ValueError:
        # int("abc") o float("0,7") fallan con un mensaje críptico; este dice qué arreglar
        kind = "entero" if cast is int else "número (con punto decimal)"
        raise ValueError(f"{name}={raw!r} en .env no es un {kind}. Ejemplo: {name}={default}") from None
    if value < minimum:
        raise ValueError(f"{name}={raw!r} en .env debe ser al menos {minimum}. Ejemplo: {name}={default}")
    return value


def load_config() -> dict[str, Any]:
    """Lee .env, lo valida y devuelve la configuración. Lanza ValueError."""
    load_dotenv()  # carga .env en el entorno; no pisa variables que ya existan
    # "true" (sin importar mayúsculas ni espacios) = LM Studio; cualquier otra cosa = OpenAI
    use_local = _env_text("USE_LOCAL_LLM", "true").lower() == "true"
    # Lo que vale igual para los dos proveedores, con su valor por defecto
    config: dict[str, Any] = {
        "use_local": use_local,
        "max_tokens": _env_number("MAX_TOKENS", "150", int, 1),  # tope de tokens por respuesta
        "temperature": _env_number("TEMPERATURE", "0.7", float, 0),  # creatividad del chat
        "timeout": _env_number("TIMEOUT", "30", float, 1),  # segundos de espera por llamada
        "max_retries": _env_number("MAX_RETRIES", "2", int, 0),  # reintentos ante fallos pasajeros
    }
    if use_local:
        # LM Studio: un servidor en tu máquina; no pide key, pero el SDK exige un texto
        config["base_url"] = _env_text("LM_STUDIO_URL", "http://localhost:1234/v1")
        config["api_key"] = "not-needed"
        # El id exacto del modelo cargado (nunca "local-model")
        config["model"] = _env_text("MODEL", "llama-3.2-3b-instruct")
        config["provider"] = "LM Studio (local)"  # el nombre que se muestra en pantalla
    else:
        # OpenAI: sin key no hay nada que hacer, así que se valida al arrancar
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            # Se detiene con los pasos para arreglarlo, antes de llamar a nadie
            raise ValueError(
                "USE_LOCAL_LLM=false pero OPENAI_API_KEY está vacía.\n"
                "  1. Pídele a tu instructor la key del curso\n"
                "  2. Agrégala a .env: OPENAI_API_KEY=sk-proj-...\n"
                "  O pon USE_LOCAL_LLM=true para usar LM Studio (gratis)."
            )
        config["base_url"] = None  # None = el default del SDK, https://api.openai.com/v1
        config["api_key"] = api_key
        config["model"] = _env_text("MODEL", "gpt-4o-mini")  # el modelo de la nube por defecto
        config["provider"] = "OpenAI (nube)"  # el nombre que se muestra en pantalla
    return config  # la configuración completa, ya validada


def check_server(config: dict[str, Any]) -> bool:
    """Solo LM Studio: GET /v1/models antes de preguntar nada."""
    # Un cliente del SDK solo para esta consulta; max_retries=0: si está apagado, lo sabes al instante
    client = openai.OpenAI(
        base_url=config["base_url"], api_key=config["api_key"], timeout=config["timeout"], max_retries=0
    )
    try:
        # GET /v1/models: la lista de modelos que LM Studio ofrece ahora mismo
        model_ids = [model.id for model in client.models.list()]
    except openai.APIConnectionError:
        # Nadie contestó en esa dirección: el servidor está apagado o la URL está mal
        print(f"❌ LM Studio no responde en {config['base_url']}")
        print("💡 Abre LM Studio, carga el modelo y enciende el servidor (pestaña Developer).")
        return False  # main() termina aquí, sin llamar al modelo
    except openai.APIStatusError as error:
        # Algo contestó, pero con error (p. ej. 404 si a la URL le falta /v1)
        print(f"❌ {config['base_url']} respondió con error {error.status_code}.")
        print("💡 Revisa LM_STUDIO_URL en .env; debe terminar en /v1 (ej. http://localhost:1234/v1).")
        return False  # main() termina aquí, sin llamar al modelo
    # Con varias variantes del mismo modelo, LM Studio agrega "@variante" al id.
    base_ids = {model_id.split("@")[0] for model_id in model_ids}
    # Si MODEL no está ni con ni sin variante, se avisa con la lista de los disponibles
    if config["model"] not in model_ids and config["model"] not in base_ids:
        print(f"❌ El modelo '{config['model']}' no está disponible en LM Studio.")
        print(f"💡 Disponibles: {', '.join(model_ids) or '(ninguno)'}. Ajusta MODEL en .env.")
        return False  # main() termina aquí, sin llamar al modelo
    return True  # servidor encendido y modelo disponible


def create_llm(config: dict[str, Any], temperature: float | None = None) -> ChatOpenAI:
    """El modelo como pieza de LangChain. temperature opcional: si no llega, la del .env."""
    # Los mismos datos que usaría el SDK, ahora en un Runnable con .invoke()
    return ChatOpenAI(
        base_url=config["base_url"],  # a qué servidor le hablas (None = OpenAI)
        api_key=config["api_key"],  # la key ("not-needed" en LM Studio)
        model=config["model"],  # el id exacto del modelo
        # La temperatura que pidió quien llama; si no pidió ninguna, la del .env
        temperature=config["temperature"] if temperature is None else temperature,
        max_tokens=config["max_tokens"],  # tope de tokens de salida por respuesta
        timeout=config["timeout"],  # segundos de espera por llamada
        max_retries=config["max_retries"],  # reintentos del SDK ante fallos pasajeros
    )