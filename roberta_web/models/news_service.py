import requests

from ..config import Config


def _normalizar_noticia(articulo):
    source = articulo.get("source", {})
    fuente = source.get("name", "") if isinstance(source, dict) else source
    return {
        "titulo": articulo.get("title", ""),
        "descripcion": articulo.get("description", ""),
        "fecha": articulo.get("publishedAt", ""),
        "fuente": fuente,
        "url": articulo.get("url", ""),
    }


def buscar_noticias(tema, cantidad=5):
    noticias = []

    if Config.NEWS_API_KEY:
        try:
            url = (
                f"https://newsapi.org/v2/everything?q={tema}&sortBy=relevancy"
                f"&pageSize={cantidad}&apiKey={Config.NEWS_API_KEY}&language=es"
            )
            response = requests.get(url, timeout=20)
            data = response.json()
            if data.get("status") == "ok" and data.get("articles"):
                noticias.extend(_normalizar_noticia(articulo) for articulo in data["articles"][:cantidad])
        except Exception:
            pass

    if not noticias and Config.GNEWS_API_KEY:
        try:
            url = f"https://gnews.io/api/v4/search?q={tema}&token={Config.GNEWS_API_KEY}&lang=es&max={cantidad}"
            response = requests.get(url, timeout=20)
            data = response.json()
            if data.get("articles"):
                noticias.extend(_normalizar_noticia(articulo) for articulo in data["articles"][:cantidad])
        except Exception:
            pass

    if not noticias:
        try:
            url = f"https://api.spaceflightnewsapi.net/v3/articles?_limit={cantidad}&title_contains={tema}"
            response = requests.get(url, timeout=20)
            data = response.json()
            noticias.extend(
                {
                    "titulo": articulo.get("title", ""),
                    "descripcion": articulo.get("summary", ""),
                    "fecha": articulo.get("publishedAt", ""),
                    "fuente": articulo.get("newsSite", ""),
                    "url": articulo.get("url", ""),
                }
                for articulo in data[:cantidad]
            )
        except Exception:
            pass

    return noticias
