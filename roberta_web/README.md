# RoBERTa Web

Aplicación web separada del notebook original, organizada con una estructura tipo MVC:

- **Modelos**: lógica de análisis, noticias, archivos e historial.
- **Controladores**: rutas Flask y orquestación de formularios.
- **Vistas**: interfaz moderna con Bootstrap 5 y estilo glassmorphism.

## Funcionalidades

- Análisis de texto con tema, sentimiento, veracidad y chatbot.
- Carga de PDF o TXT para extraer texto y analizarlo.
- Búsqueda de noticias desde APIs externas.
- Verificación de noticias individuales.
- Análisis masivo de carpetas de noticias verdaderas y falsas.
- Guardado de respuestas en historial JSON.

## Requisitos

Instala dependencias desde la carpeta raíz del proyecto:

```bash
pip install -r roberta_web/requirements.txt
```

## Variables de entorno

- `OPENAI_API_KEY`
- `NEWS_API_KEY`
- `GNEWS_API_KEY`
- `FLASK_SECRET_KEY` opcional

## Ejecución

Desde la raíz del repositorio:

```bash
python -m roberta_web.app
```

O usa Flask:

```bash
flask --app roberta_web.app run
```

## Notas

- Si no hay clave de OpenAI, las funciones que dependen de GPT devolverán mensajes de error controlados.
- Si faltan Torch o Transformers, la app cae a modo básico para tema y sentimiento.
- El análisis masivo usa rutas locales del equipo, igual que el notebook original.
