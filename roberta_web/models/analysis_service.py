import os
import re
from functools import lru_cache

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
except Exception:
    RobertaForSequenceClassification = None
    RobertaTokenizer = None
    pipeline = None


DEFAULT_TOPICS = [
    "COVID-19",
    "Salud",
    "Política",
    "Economía",
    "Tecnología",
    "Ciencia",
    "Deportes",
    "Entretenimiento",
    "Educación",
    "Medio Ambiente",
]

POSITIVE_WORDS = ["bueno", "excelente", "genial", "increíble", "positivo", "éxito", "feliz", "alegre", "satisfecho"]
NEGATIVE_WORDS = ["malo", "terrible", "horrible", "negativo", "fracaso", "triste", "infeliz", "insatisfecho"]

_TRANSFORMERS_READY = None
_TRANSFORMER_TOKENIZER = None
_TRANSFORMER_MODEL = None
_ZERO_SHOT_CLASSIFIER = None


def _recortar_texto(texto, limite):
    if not texto:
        return ""
    return texto[:limite] if len(texto) > limite else texto


def _api_key(nombre_env):
    return os.getenv(nombre_env, "").strip()


@lru_cache(maxsize=1)
def _get_openai_client():
    api_key = _api_key("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def _llamar_chat_openai(system_content, user_content, max_tokens=300, temperature=0.3, model="gpt-4o-mini"):
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY no configurada o librería openai no disponible")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _ensure_transformers():
    global _TRANSFORMERS_READY, _TRANSFORMER_TOKENIZER, _TRANSFORMER_MODEL, _ZERO_SHOT_CLASSIFIER

    if _TRANSFORMERS_READY is not None:
        return _TRANSFORMERS_READY

    _TRANSFORMERS_READY = False
    if RobertaTokenizer is None or RobertaForSequenceClassification is None or pipeline is None:
        return _TRANSFORMERS_READY

    if np is not None:
        version = getattr(np, "__version__", "")
        if version.startswith("2."):
            return _TRANSFORMERS_READY

    try:
        _TRANSFORMER_TOKENIZER = RobertaTokenizer.from_pretrained("roberta-base")
        _TRANSFORMER_MODEL = RobertaForSequenceClassification.from_pretrained("roberta-base")
        _ZERO_SHOT_CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        _TRANSFORMERS_READY = True
    except Exception:
        _TRANSFORMER_TOKENIZER = None
        _TRANSFORMER_MODEL = None
        _ZERO_SHOT_CLASSIFIER = None
        _TRANSFORMERS_READY = False

    return _TRANSFORMERS_READY


def usar_transformers():
    return _ensure_transformers()


def detectar_tema(texto, temas_posibles=None):
    if not temas_posibles:
        temas_posibles = list(DEFAULT_TOPICS)

    texto_corto = _recortar_texto(texto, 1000)

    if not usar_transformers():
        tema_detectado = "General"
        max_coincidencias = 0
        confianza = 0.5

        texto_minuscula = texto_corto.lower()
        for tema in temas_posibles:
            palabras_tema = tema.lower().split()
            coincidencias = 0
            for palabra in palabras_tema:
                if palabra in texto_minuscula:
                    coincidencias += texto_minuscula.count(palabra)
            if coincidencias > max_coincidencias:
                max_coincidencias = coincidencias
                tema_detectado = tema
                confianza = min(0.5 + (coincidencias / 10), 0.95)

        return tema_detectado, confianza

    try:
        resultado = _ZERO_SHOT_CLASSIFIER(texto_corto, temas_posibles)
        return resultado["labels"][0], resultado["scores"][0]
    except Exception:
        return "General", 0.5


def analizar_sentimiento(texto):
    if not usar_transformers():
        texto_lower = texto.lower()
        count_pos = sum(texto_lower.count(p) for p in POSITIVE_WORDS)
        count_neg = sum(texto_lower.count(n) for n in NEGATIVE_WORDS)

        if count_pos > count_neg:
            return "positivo", 0.5 + min(count_pos / (count_pos + count_neg + 1), 0.45)
        if count_neg > count_pos:
            return "negativo", 0.5 + min(count_neg / (count_pos + count_neg + 1), 0.45)
        return "neutral", 0.5

    try:
        inputs = _TRANSFORMER_TOKENIZER(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = _TRANSFORMER_MODEL(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confianza = probs[0, pred].item()

        sentimiento = "neutral"
        if pred == 0:
            sentimiento = "positivo"
        elif pred == 1:
            sentimiento = "negativo"

        return sentimiento, confianza
    except Exception:
        return "neutral", 0.5


def _extraer_puntuacion_veracidad(analisis):
    patron = r"(?:puntuaci[oó]n|veracidad).*?(?:de|es).*?(\d+)"
    match = re.search(patron, analisis.lower())
    return int(match.group(1)) if match else 50


def _clasificar_veracidad(puntuacion):
    if puntuacion >= 80:
        return "Verdadero"
    if puntuacion >= 60:
        return "Mayormente verdadero"
    if puntuacion >= 40:
        return "Parcialmente verdadero"
    if puntuacion >= 20:
        return "Mayormente falso"
    return "Falso"


def _extraer_listado_numerado(texto):
    elementos = []
    for linea in texto.strip().split("\n"):
        match = re.match(r"^\d+\.\s*(.+)$", linea.strip())
        if match:
            elementos.append(match.group(1).strip())
    return elementos


def verificar_veracidad(texto, tema=None):
    if not texto:
        return {"veracidad": "No determinable", "confianza": 0, "razones": ["Texto vacío"], "fuentes_contradictorias": []}

    texto_analizar = _recortar_texto(texto, 4000)
    prompt = (
        "Analiza el siguiente texto y determina su veracidad. \n"
        "    Identifica afirmaciones falsas, engañosas o inexactas. \n"
        "    Proporciona un análisis detallado que incluya:\n"
        "    1. Si el texto contiene información falsa o engañosa\n"
        "    2. Qué afirmaciones específicas son falsas o engañosas\n"
        "    3. La explicación de por qué son falsas\n"
        "    4. Cualquier contradicción interna en el texto\n"
        "    5. Una puntuación de veracidad del 0 al 100\n\n"
        "    Texto para analizar:\n"
    )
    if tema:
        prompt += f"\nTema: {tema}\n"
    prompt += f"\n{texto_analizar}"

    try:
        analisis = _llamar_chat_openai(
            system_content=(
                "Eres un verificador de hechos experto. Tu tarea es analizar textos para determinar su veracidad, "
                "identificar información falsa o engañosa, y explicar por qué. Tus respuestas deben seguir un formato "
                "estructurado que incluya: 1) Veredicto de veracidad, 2) Afirmaciones falsas identificadas, 3) Explicación "
                "de cada falsedad, 4) Puntuación de veracidad."
            ),
            user_content=prompt,
            max_tokens=800,
            temperature=0.3,
        )
        puntuacion = _extraer_puntuacion_veracidad(analisis)
        categoria = _clasificar_veracidad(puntuacion)
        match_afirmaciones = re.search(
            r"(?:afirmaci[oó]n(?:es)? falsa(?:s)?|informaci[oó]n falsa)[:\s]*([^\n.]*)",
            analisis.lower(),
        )
        afirmaciones_falsas = []
        if match_afirmaciones:
            afirmaciones_falsas = [a.strip() for a in re.split(r"[\n\d\.\-•]+", match_afirmaciones.group(1)) if a.strip()]
        if not afirmaciones_falsas and puntuacion < 60:
            afirmaciones_falsas = ["El análisis detectó contenido cuestionable pero no especificó afirmaciones concretas"]
        return {
            "veracidad": categoria,
            "confianza": puntuacion / 100,
            "razones": afirmaciones_falsas,
            "analisis_completo": analisis,
        }
    except Exception as e:
        return {"veracidad": "Error en análisis", "confianza": 0, "razones": [f"Error: {e}"], "analisis_completo": ""}


def extraer_afirmaciones(texto, cantidad=5):
    if not texto or len(texto) < 50:
        return []

    texto_corto = _recortar_texto(texto, 3000)
    try:
        resultado = _llamar_chat_openai(
            system_content=(
                "Extrae las principales afirmaciones verificables del siguiente texto. Devuelve SOLO una lista numerada de "
                "las 5 afirmaciones más importantes o controvertidas que requieren verificación de hechos. No incluyas "
                "opiniones o juicios de valor."
            ),
            user_content=texto_corto,
            max_tokens=300,
            temperature=0.3,
        )
        return _extraer_listado_numerado(resultado)[:cantidad]
    except Exception:
        return []


def verificar_afirmacion(afirmacion, tema=None):
    if not afirmacion:
        return {"veracidad": "No determinable", "explicacion": "Afirmación vacía"}

    try:
        prompt = "Verifica la siguiente afirmación y determina si es verdadera, falsa o no verificable."
        if tema:
            prompt += f"\n\nContexto: La afirmación está relacionada con el tema de '{tema}'."
        prompt += f"\n\nAfirmación: '{afirmacion}'"
        resultado = _llamar_chat_openai(
            system_content=(
                "Eres un verificador de hechos experto. Tu tarea es analizar una afirmación y determinar su veracidad "
                "basándote en hechos verificables. Da un veredicto claro (Verdadero/Falso/Parcialmente verdadero/No "
                "verificable) seguido de una explicación concisa."
            ),
            user_content=prompt,
            max_tokens=300,
            temperature=0.3,
        )
        match = re.search(r"(verdadero|falso|parcialmente verdadero|no verificable)", resultado.lower())
        veredicto = "No determinable"
        if match:
            t = match.group(1)
            if "falso" in t and "parcialmente" not in t:
                veredicto = "Falso"
            elif "verdadero" in t and "parcialmente" not in t:
                veredicto = "Verdadero"
            elif "parcialmente" in t:
                veredicto = "Parcialmente verdadero"
            elif "no verificable" in t:
                veredicto = "No verificable"
        return {"veracidad": veredicto, "explicacion": resultado}
    except Exception as e:
        return {"veracidad": "Error en análisis", "explicacion": f"Error: {e}"}


def chatbot_tematico(pregunta, tema="General", contexto_texto=None, noticias=None, verificar=False):
    prompt = f"Tema: {tema}\n"
    if contexto_texto:
        contexto_resumido = _recortar_texto(contexto_texto, 2000)
        prompt += f"Contexto del documento:\n{contexto_resumido}\n\n"
    if noticias:
        prompt += "Noticias recientes sobre este tema:\n"
        for i, noticia in enumerate(noticias[:3], 1):
            prompt += f"{i}. {noticia['titulo']} - {noticia['descripcion']}\n"
        prompt += "\n"

    system_content = f"Eres un asistente experto en {tema}. Proporciona información precisa basada en los datos disponibles."
    if verificar:
        system_content = (
            f"Eres un asistente experto en {tema} con capacidades de verificación de hechos. Analiza críticamente la "
            "información proporcionada. Identifica posibles afirmaciones falsas o engañosas y verifica su veracidad antes de "
            "responder. Al responder, indica claramente qué información es verificable y cuál podría ser cuestionable."
        )
        prompt += "\nIMPORTANTE: Verifica la veracidad de tu respuesta. Indica explícitamente cuando la información pueda ser cuestionable o no verificable."
    prompt += f"\nPregunta: {pregunta}"

    try:
        return _llamar_chat_openai(system_content=system_content, user_content=prompt, max_tokens=500, temperature=0.7)
    except Exception as e:
        return f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
