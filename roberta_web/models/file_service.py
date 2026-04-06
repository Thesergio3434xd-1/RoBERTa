import glob
import json
import os
from datetime import datetime

try:
    import fitz
except Exception:
    fitz = None

from .analysis_service import detectar_tema, verificar_veracidad


def extraer_texto_pdf(path_pdf):
    if fitz is None:
        return ""
    try:
        doc = fitz.open(path_pdf)
        texto = ""
        for pagina in doc:
            texto += pagina.get_text()
        return texto
    except Exception:
        return ""


def leer_archivo(ruta):
    try:
        if ruta.lower().endswith(".pdf"):
            return extraer_texto_pdf(ruta)
        with open(ruta, "r", encoding="utf-8") as archivo:
            return archivo.read()
    except Exception:
        return ""


def analizar_masivo_archivos(directorio_verdaderas, directorio_falsas, max_archivos=10):
    resultados = {
        "verdaderas": {"archivos": [], "correctos": 0, "incorrectos": 0, "detalles": []},
        "falsas": {"archivos": [], "correctos": 0, "incorrectos": 0, "detalles": []},
        "metricas": {},
        "confusion_matrix": {"VP": 0, "VN": 0, "FP": 0, "FN": 0},
    }

    archivos_verdaderos = glob.glob(os.path.join(directorio_verdaderas, "*"))[:max_archivos]
    archivos_falsos = glob.glob(os.path.join(directorio_falsas, "*"))[:max_archivos]

    for archivo in archivos_verdaderos:
        contenido = leer_archivo(archivo)
        if not contenido:
            continue
        tema_detectado, conf_tema = detectar_tema(contenido)
        resultado_verificacion = verificar_veracidad(contenido, tema=tema_detectado)
        es_correcto = resultado_verificacion["confianza"] >= 0.6
        detalle = {
            "archivo": os.path.basename(archivo),
            "tema": tema_detectado,
            "confianza_tema": conf_tema,
            "veracidad": resultado_verificacion["veracidad"],
            "confianza_veracidad": resultado_verificacion["confianza"],
            "clasificacion_correcta": es_correcto,
            "contenido_preview": contenido[:200] + "..." if len(contenido) > 200 else contenido,
        }
        resultados["verdaderas"]["detalles"].append(detalle)
        if es_correcto:
            resultados["verdaderas"]["correctos"] += 1
            resultados["confusion_matrix"]["VP"] += 1
        else:
            resultados["verdaderas"]["incorrectos"] += 1
            resultados["confusion_matrix"]["FN"] += 1

    for archivo in archivos_falsos:
        contenido = leer_archivo(archivo)
        if not contenido:
            continue
        tema_detectado, conf_tema = detectar_tema(contenido)
        resultado_verificacion = verificar_veracidad(contenido, tema=tema_detectado)
        es_correcto = resultado_verificacion["confianza"] < 0.6
        detalle = {
            "archivo": os.path.basename(archivo),
            "tema": tema_detectado,
            "confianza_tema": conf_tema,
            "veracidad": resultado_verificacion["veracidad"],
            "confianza_veracidad": resultado_verificacion["confianza"],
            "clasificacion_correcta": es_correcto,
            "contenido_preview": contenido[:200] + "..." if len(contenido) > 200 else contenido,
        }
        resultados["falsas"]["detalles"].append(detalle)
        if es_correcto:
            resultados["falsas"]["correctos"] += 1
            resultados["confusion_matrix"]["VN"] += 1
        else:
            resultados["falsas"]["incorrectos"] += 1
            resultados["confusion_matrix"]["FP"] += 1

    cm = resultados["confusion_matrix"]
    total = cm["VP"] + cm["VN"] + cm["FP"] + cm["FN"]
    if total > 0:
        precision = cm["VP"] / (cm["VP"] + cm["FP"]) if (cm["VP"] + cm["FP"]) > 0 else 0
        recall = cm["VP"] / (cm["VP"] + cm["FN"]) if (cm["VP"] + cm["FN"]) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (cm["VP"] + cm["VN"]) / total
        resultados["metricas"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "total_archivos": total,
        }

    return resultados


def guardar_respuesta(tema, pregunta, respuesta, verificacion=None, historial_folder=None):
    historial_folder = historial_folder or os.path.join(os.path.dirname(__file__), "..", "data", "historial")
    historial_folder = os.path.abspath(historial_folder)
    os.makedirs(historial_folder, exist_ok=True)

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registro = {
        "fecha": fecha,
        "tema": tema,
        "pregunta": pregunta,
        "respuesta": respuesta,
    }
    if verificacion:
        registro["verificacion"] = verificacion

    filename = os.path.join(historial_folder, f"historial_{datetime.now().strftime('%Y%m%d')}.json")
    historial = []
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as archivo:
                historial = json.load(archivo)
        except Exception:
            historial = []

    historial.append(registro)
    with open(filename, "w", encoding="utf-8") as archivo:
        json.dump(historial, archivo, ensure_ascii=False, indent=2)

    return filename
