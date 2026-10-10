"""
fallback_demo.py: si el modelo principal falla, LangChain prueba el de respaldo.
Aquí el principal apunta a un puerto donde no hay nada, para ver el mecanismo sin gastar.
"""

# Lo tuyo: leer la configuración y crear modelos con ella
from aurora_llm import create_llm, load_config

config = load_config()  # la configuración de siempre, desde .env
# Una copia de la configuración que apunta a un puerto sin servidor, y sin reintentos
broken = dict(config, base_url="http://localhost:9999/v1", max_retries=0)
# Si el principal lanza una excepción, with_fallbacks prueba el modelo de siempre
llm = create_llm(broken).with_fallbacks([create_llm(config)])
answer = llm.invoke("Say hello to the Café Aurora staff in five words.")  # una sola llamada, con respaldo
print(f"🤖 {answer.content}")  # la respuesta, venga de quien venga
print(f"   respondió: {answer.response_metadata.get('model_name')}")  # el modelo que contestó: el de respaldo, porque en el 9999 no hay nada