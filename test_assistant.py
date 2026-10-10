"""
test_assistant.py: las piezas del asistente, probadas sin servidor y sin gastar.
Se corre con: python test_assistant.py
"""

# FakeListChatModel: un modelo falso que devuelve respuestas fijas, sin servidor
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

# Las piezas tuyas que se prueban
from prompts import CHAT_PROMPT, REPLY_PROMPT, ReviewData
from week3_assistant import MAX_HISTORY_MESSAGES


def test_prompt_variables():
    # Las casillas que espera cada prompt (LangChain las ordena alfabéticamente)
    assert CHAT_PROMPT.input_variables == ["history", "question"]
    assert REPLY_PROMPT.input_variables == ["product", "rating", "review"]


def test_history_goes_between_system_and_question():
    # Un turno anterior en el historial; el prompt debe ponerlo entre el sistema y la pregunta
    history = [HumanMessage("I am Daniela."), AIMessage("Hi Daniela.")]
    # Llena el prompt sin llamar al modelo y saca la lista de mensajes
    messages = CHAT_PROMPT.invoke({"history": history, "question": "Who am I?"}).to_messages()
    # El orden de los roles: sistema, el turno guardado y la pregunta nueva
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]


def test_chat_chain_returns_text():
    # Un modelo falso que siempre contesta lo mismo: prueba que la cadena está bien conectada
    fake_llm = FakeListChatModel(responses=["You are Daniela."])
    chain = CHAT_PROMPT | fake_llm | StrOutputParser()  # la misma forma que la cadena del chat
    # Si la cadena está bien armada, la respuesta del falso sale intacta al final
    assert chain.invoke({"history": [], "question": "Who am I?"}) == "You are Daniela."


def test_trim_keeps_last_messages_starting_on_human():
    # Seis turnos = 12 mensajes; la ventana deja los últimos 10, empezando en una pregunta
    history = []
    for turn in range(6):
        history += [HumanMessage(f"question {turn}"), AIMessage(f"answer {turn}")]
    # El mismo recorte que usa run_chat()
    trimmer = trim_messages(
        max_tokens=MAX_HISTORY_MESSAGES, token_counter=len, strategy="last", start_on="human"
    )
    recent = trimmer.invoke(history)
    # Quedan 10 mensajes, y el primero es la pregunta del turno 1 (el turno 0 se recortó)
    assert len(recent) == MAX_HISTORY_MESSAGES
    assert recent[0].type == "human" and recent[0].content == "question 1"


def test_review_data_rejects_bad_rating():
    # Un dato bueno pasa (sin precio: queda en None)
    assert ReviewData.model_validate({"product": "cold brew", "rating": 2}).price_mxn is None
    # Una calificación en texto tiene que fallar con ValidationError
    try:
        ReviewData.model_validate({"product": "latte", "rating": "four"})
    except ValidationError:
        return  # falló como debía: la prueba pasa
    raise AssertionError("rating='four' should fail")  # si llegó aquí, la validación no funcionó


if __name__ == "__main__":
    # Sin pytest: se llaman a mano; si una falla, verás su AssertionError
    test_prompt_variables()  # los prompts
    test_history_goes_between_system_and_question()  # el lugar del historial
    test_chat_chain_returns_text()  # la cadena del chat
    test_trim_keeps_last_messages_starting_on_human()  # la ventana
    test_review_data_rejects_bad_rating()  # la validación de ReviewData
    print("5 pruebas OK")  # solo se imprime si las cinco pasaron