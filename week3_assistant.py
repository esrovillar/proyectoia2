"""
week3_assistant.py: el asistente del personal de Café Aurora, ahora con LangChain.

Funciona con LM Studio (local) u OpenAI (nube): USE_LOCAL_LLM en .env decide.
    python week3_assistant.py          chat con memoria (escribe "salir" para terminar)
    python week3_assistant.py reply    pegas una reseña, recibes datos y una respuesta
"""

# sys: los argumentos de la línea de comandos y el código de salida; Any: para tipar la config
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import openai  # las excepciones del SDK; las de ChatOpenAI heredan de ellas
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages  # historial y recorte
from langchain_core.output_parsers import StrOutputParser  # saca el texto del AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # para conectar pasos
from langchain_openai.chat_models.base import OpenAIRefusalError  # el modelo se negó a contestar
from pydantic import ValidationError  # el JSON no cumple ReviewData

# Lo tuyo: la configuración y el modelo, y los prompts
from aurora_llm import check_server, create_llm, load_config
from prompts import CHAT_PROMPT, EXTRACT_PROMPT, REPLY_PROMPT, ReviewData

MAX_HISTORY_MESSAGES = 10  # los últimos 10 mensajes: 5 preguntas y 5 respuestas

def explain_error(error: Exception, config: dict[str, Any]) -> None:
    """Los fallos de la Week 2, más los de la salida estructurada. Lo que no reconoce, lo relanza."""
    # El orden importa: cada hija va antes que su madre (APITimeoutError es un APIConnectionError)
    if isinstance(error, openai.APITimeoutError):  # el servidor no contestó a tiempo
        print(f"❌ Sin respuesta tras {config['timeout']:g} s (y {config['max_retries']} reintentos).")
    elif isinstance(error, openai.APIConnectionError):  # no hubo con quién hablar
        print(f"❌ No se pudo conectar con {config['provider']}.")
    elif isinstance(error, openai.AuthenticationError):  # 401: key inválida o revocada
        print("❌ 401: la API key fue rechazada. Pídele otra a tu instructor.")
    elif isinstance(error, openai.RateLimitError):  # 429: demasiadas llamadas o sin saldo
        print(f"❌ 429 de {config['provider']}: {error.code or 'rate limit'}.")
    elif isinstance(error, openai.APIStatusError):  # cualquier otro código HTTP de error
        print(f"❌ HTTP {error.status_code} de {config['provider']}: {error.message}")
    elif isinstance(error, openai.LengthFinishReasonError):  # el JSON se cortó en max_tokens
        print(f"❌ El JSON se cortó al llegar a MAX_TOKENS ({config['max_tokens']}). Súbelo en tu .env.")
    elif isinstance(error, OpenAIRefusalError):  # el modelo se negó a contestar
        print(f"❌ El modelo se negó a extraer los datos: {error}")
    elif isinstance(error, ValidationError):  # el JSON llegó, pero no cumple ReviewData
        print(f"❌ El JSON del modelo no cumple ReviewData: {error.errors()[0]['msg']}")
    else:
        raise error  # un error de programación tiene que verse completo


def run_chat(llm, config: dict[str, Any]) -> int:
    """Modo chat: la memoria es una lista de mensajes que tú guardas y reenvías."""
    chain = CHAT_PROMPT | llm  # sin StrOutputParser: queremos el AIMessage para leer los tokens
    # Recorta el historial a los últimos N mensajes; token_counter=len cuenta mensajes, no tokens
    trimmer = trim_messages(
        max_tokens=MAX_HISTORY_MESSAGES, token_counter=len, strategy="last", start_on="human"
    )
    history: list[BaseMessage] = []  # toda la conversación; vive mientras corre el programa
    print('💬 Escribe tu pregunta ("salir" para terminar).')
    while True:  # un turno por vuelta, hasta que la persona salga
        try:
            question = input("💬 Tú: ").strip()  # lee la pregunta y le quita espacios
        except (EOFError, KeyboardInterrupt):  # Ctrl+D o Ctrl+C también cierran el chat
            print()  # salto de línea, para que la terminal no quede pegada al prompt
            return 0
        if question.lower() in ("salir", "exit"):  # la palabra para terminar
            return 0
        if not question:  # Enter sin texto: no se gasta una llamada, se vuelve a preguntar
            continue
        recent = trimmer.invoke(history)  # lo que viaja: solo la ventana reciente
        try:
            # El prompt recibe el historial recortado y la pregunta; el modelo contesta
            answer = chain.invoke({"history": recent, "question": question})
        except (openai.OpenAIError, OpenAIRefusalError, ValidationError) as error:
            explain_error(error, config)  # explica el fallo y deja preguntar otra vez
            continue
        history += [HumanMessage(question), AIMessage(answer.content)]  # se guarda el turno completo
        usage = answer.usage_metadata or {}  # los tokens de esta llamada (vacío si el servidor no los manda)
        print(f"🤖 {answer.content}")  # la respuesta
        # El costo de la memoria a la vista: tokens de entrada, de salida y cuántos mensajes viajaron
        print(f"   📊 tokens entrada {usage.get('input_tokens', 0)} / salida {usage.get('output_tokens', 0)} · historial {len(history)} mensajes (se enviaron {len(recent)})")

def build_reply_pipeline(config: dict[str, Any]):
    """Reseña → ReviewData → respuesta. Dos llamadas al modelo en una sola cadena."""
    # Paso 1, temperatura 0: el dato, sin creatividad; devuelve un ReviewData validado
    extract = EXTRACT_PROMPT | create_llm(config, temperature=0).with_structured_output(ReviewData)
    # Adaptador: de la hoja {"review", "data"} a las tres variables de REPLY_PROMPT
    to_reply_input = RunnableLambda(
        lambda x: {"review": x["review"], "product": x["data"].product, "rating": x["data"].rating}
    )
    # Paso 2, temperatura 0.3: un poco de variación en el texto, sin inventar
    draft = REPLY_PROMPT | create_llm(config, temperature=0.3) | StrOutputParser()
    # assign deja pasar la entrada y le agrega una llave: primero "data", después "reply"
    return RunnablePassthrough.assign(data=extract) | RunnablePassthrough.assign(reply=to_reply_input | draft)

def run_reply(config: dict[str, Any]) -> int:
    """Modo reply: la reseña pasa por dos pasos encadenados con LCEL."""
    review = input("📝 Pega una reseña: ").strip()  # lee la reseña completa
    if not review:  # sin reseña no hay nada que mandar al modelo
        print("❌ Reseña vacía: no hay nada que enviar.")
        return 1  # código 1: el programa terminó con un problema
    print(f"🔄 Procesando con {config['provider']}...")  # aviso: las dos llamadas tardan
    try:
        # La cadena completa: extraer los datos y redactar la respuesta, en ese orden
        result = build_reply_pipeline(config).invoke({"review": review})
    except (openai.OpenAIError, OpenAIRefusalError, ValidationError) as error:
        explain_error(error, config)  # explica el fallo y termina con código 1
        return 1
    # result trae todo lo que se fue agregando: review, data y reply
    print(f"✅ ReviewData: {result['data'].model_dump()}")  # los datos, como diccionario
    print(f"✉️  Respuesta sugerida: {result['reply']}")  # el texto para publicar
    return 0  # código 0: todo salió bien

def main() -> int:
    """Carga la config, verifica el servidor y corre el modo pedido."""
    print("🚀 Week 3 · Asistente de Café Aurora con LangChain")
    try:
        config = load_config()  # lee y valida el .env
    except ValueError as error:  # falta la key: se avisa antes de llamar a nadie
        print(f"❌ Error de configuración:\n{error}")
        return 1
    print(f"✅ Proveedor: {config['provider']} · modelo: {config['model']}")
    if config["use_local"] and not check_server(config):  # solo LM Studio se verifica antes
        return 1
    if sys.argv[1:] == ["reply"]:  # python week3_assistant.py reply
        return run_reply(config)
    return run_chat(create_llm(config), config)  # sin argumentos: el chat, con el modelo del .env




if __name__ == "__main__":
    sys.exit(main())  # el número que devuelve main() es el código de salida del programa