import os

from flask import Blueprint, flash, render_template, request

from ..config import Config
from ..models.analysis_service import (
    analizar_sentimiento,
    chatbot_tematico,
    detectar_tema,
    extraer_afirmaciones,
    usar_transformers,
    verificar_afirmacion,
    verificar_veracidad,
)
from ..models.file_service import analizar_masivo_archivos, extraer_texto_pdf, guardar_respuesta
from ..models.news_service import buscar_noticias


main_bp = Blueprint("main", __name__)


def _leer_archivo_subido(archivo):
    nombre_seguro = archivo.filename.replace("..", "_")
    ruta_salida = os.path.join(Config.UPLOAD_FOLDER, nombre_seguro)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    archivo.save(ruta_salida)

    if ruta_salida.lower().endswith(".pdf"):
        return extraer_texto_pdf(ruta_salida)

    with open(ruta_salida, "r", encoding="utf-8", errors="ignore") as archivo_txt:
        return archivo_txt.read()


def _analizar_contenido(texto, tema_preferido="", pregunta="", verificar=False):
    tema_detectado, confianza_tema = detectar_tema(texto)
    tema_final = tema_preferido or tema_detectado
    sentimiento, confianza_sentimiento = analizar_sentimiento(texto[:512])
    noticias = buscar_noticias(tema_final, 3)
    resultado = {
        "tema_detectado": tema_detectado,
        "confianza_tema": confianza_tema,
        "sentimiento": sentimiento,
        "confianza_sentimiento": confianza_sentimiento,
        "verificacion": verificar_veracidad(texto, tema=tema_final),
        "afirmaciones": extraer_afirmaciones(texto),
        "noticias": noticias,
    }
    respuesta_chat = None
    if pregunta:
        respuesta_chat = chatbot_tematico(
            pregunta,
            tema=tema_final,
            contexto_texto=texto,
            noticias=noticias,
            verificar=verificar,
        )
        guardar_respuesta(tema_final, pregunta, respuesta_chat, verificacion=resultado["verificacion"]["veracidad"])
    return resultado, respuesta_chat


@main_bp.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        transformers_activos=usar_transformers(),
        openai_listo=bool(Config.OPENAI_API_KEY),
        noticias_listas=bool(Config.NEWS_API_KEY or Config.GNEWS_API_KEY),
    )


@main_bp.route("/analizar-texto", methods=["GET", "POST"])
def analizar_texto():
    resultado = None
    tema = ""
    texto = ""
    pregunta = ""
    respuesta_chat = None

    if request.method == "POST":
        texto = request.form.get("texto", "").strip()
        tema = request.form.get("tema", "").strip()
        pregunta = request.form.get("pregunta", "").strip()
        verificar = request.form.get("verificar") == "on"

        if not texto:
            flash("Ingresa un texto para analizar.", "warning")
        else:
            resultado, respuesta_chat = _analizar_contenido(texto, tema_preferido=tema, pregunta=pregunta, verificar=verificar)

    return render_template("text_analysis.html", resultado=resultado, texto=texto, tema=tema, pregunta=pregunta, respuesta_chat=respuesta_chat)


@main_bp.route("/analizar-pdf", methods=["GET", "POST"])
def analizar_pdf():
    resultado = None
    texto_extraido = ""
    tema = ""
    pregunta = ""
    respuesta_chat = None

    if request.method == "POST":
        archivo = request.files.get("archivo_pdf")
        tema = request.form.get("tema", "").strip()
        pregunta = request.form.get("pregunta", "").strip()
        verificar = request.form.get("verificar") == "on"

        if not archivo or not archivo.filename:
            flash("Selecciona un archivo PDF o TXT.", "warning")
        else:
            texto_extraido = _leer_archivo_subido(archivo)

            if not texto_extraido:
                flash("No se pudo extraer texto del archivo.", "danger")
            else:
                resultado, respuesta_chat = _analizar_contenido(texto_extraido, tema_preferido=tema, pregunta=pregunta, verificar=verificar)

    return render_template("pdf_analysis.html", resultado=resultado, texto_extraido=texto_extraido, tema=tema, pregunta=pregunta, respuesta_chat=respuesta_chat)


@main_bp.route("/buscar-noticias", methods=["GET", "POST"])
def buscar_noticias_view():
    noticias = []
    noticia_verificada = None
    tema = ""
    cantidad = 5

    if request.method == "POST":
        tema = request.form.get("tema", "").strip()
        cantidad = request.form.get("cantidad", 5, type=int)
        verificacion_index = request.form.get("verificacion_index", "")

        if tema:
            noticias = buscar_noticias(tema, max(1, min(cantidad, 10)))
            if verificacion_index.isdigit():
                indice = int(verificacion_index)
                if 0 <= indice < len(noticias):
                    noticia = noticias[indice]
                    noticia_verificada = verificar_veracidad(
                        f"Título: {noticia['titulo']}\n\nDescripción: {noticia['descripcion']}\n\nFuente: {noticia['fuente']}\nFecha: {noticia['fecha']}",
                        tema=tema,
                    )
        else:
            flash("Ingresa un tema para buscar noticias.", "warning")

    return render_template("news_search.html", noticias=noticias, noticia_verificada=noticia_verificada, tema=tema, cantidad=cantidad)


@main_bp.route("/analisis-masivo", methods=["GET", "POST"])
def analisis_masivo():
    resultados = None
    dir_verdaderas = ""
    dir_falsas = ""
    max_archivos = 10

    if request.method == "POST":
        dir_verdaderas = request.form.get("dir_verdaderas", "").strip().strip('"')
        dir_falsas = request.form.get("dir_falsas", "").strip().strip('"')
        max_archivos = request.form.get("max_archivos", 10, type=int)
        max_archivos = max(1, min(max_archivos, 10))

        if not os.path.exists(dir_verdaderas):
            flash(f"No se encontró el directorio de verdaderas: {dir_verdaderas}", "danger")
        elif not os.path.exists(dir_falsas):
            flash(f"No se encontró el directorio de falsas: {dir_falsas}", "danger")
        else:
            resultados = analizar_masivo_archivos(dir_verdaderas, dir_falsas, max_archivos)

    return render_template(
        "batch_analysis.html",
        resultados=resultados,
        dir_verdaderas=dir_verdaderas,
        dir_falsas=dir_falsas,
        max_archivos=max_archivos,
    )
