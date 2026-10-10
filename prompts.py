"""
prompts.py: los prompts de Café Aurora como objetos, separados del código que los usa.
Las instrucciones que lee el modelo van en inglés; las variables van entre llaves.
"""

# ChatPromptTemplate: un prompt con roles y variables, que se llena con .invoke()
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # MessagesPlaceholder: el hueco del historial
from pydantic import BaseModel, Field  # para describir la forma de los datos de una reseña

# Qué sabe y qué no sabe el asistente, y que recuerde lo que le dicen en la conversación
SYSTEM_PROMPT = (
    "You are an assistant for the staff of Café Aurora, a coffee shop. "
    "Answer general questions accurately, in the language of the question, in at most three sentences. "
    "You do not know Café Aurora's menu, prices, hours or policies; if asked, say you do not have that information yet. "
    "Remember what the user tells you in this conversation, such as their name or role, and use it when they ask."
)

# El chat: sistema, después el historial (lo llena tu código) y al final la pregunta nueva
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),  # las instrucciones fijas
    MessagesPlaceholder("history"),  # aquí entran los mensajes anteriores, en orden
    ("human", "{question}"),  # la pregunta de este turno, en la variable {question}
])

# Paso 1 del modo reply: sacar los datos de la reseña
EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract the data from this Café Aurora customer review."),  # qué hacer
    ("human", "{review}"),  # la reseña, tal como la pegó Rodrigo
])

# Paso 2 del modo reply: redactar la respuesta. {product} va en el sistema para que no invente otro
REPLY_PROMPT = ChatPromptTemplate.from_messages([
    # Las reglas de la respuesta, con el producto exacto y la disculpa si la calificación es baja
    ("system",
     "You write replies from Café Aurora to customer reviews. "
     "Reply in the language of the review, in at most two sentences. "
     "Thank the customer and mention the product exactly as written: {product}. Do not mention any other product. "
     "If the rating is 3 or lower, apologize and invite them back."),
    # La reseña original y la calificación que sacó el paso 1
    ("human", "Review: {review}\nRating: {rating}"),
])

class ReviewData(BaseModel):
    """The data Café Aurora wants from each customer review."""

    # Cada description viaja al modelo dentro del JSON Schema: es parte del prompt
    product: str = Field(description="The product the review talks about")  # obligatorio
    # Opcional: si la reseña no dice precio, queda en None
    price_mxn: float | None = Field(default=None, description="The price in Mexican pesos, if it is mentioned")
    rating: int = Field(description="The rating from 1 to 5, where 5 is the best")  # entero obligatorio