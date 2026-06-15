# 📚 Sistema de Gestión de Documentos Académicos con RAG

> **Proyecto Integrador — Segundo Parcial**
> Tópicos Especiales I · Universidad Tecnológica de Panamá
> **Grupo 7**

Sistema que permite **cargar documentos académicos (PDFs)**, **indexarlos con
embeddings** en una base de datos vectorial y **responder preguntas sobre su
contenido** mediante **RAG** (Retrieval-Augmented Generation), todo desde un
**dashboard interactivo** en Streamlit.

---

## 🎯 Cumplimiento de requisitos

### Requisitos generales del Segundo Parcial

| Requisito | Cómo se cumple |
|-----------|----------------|
| **Pipeline de datos (ingesta de PDFs)** | PDFs subidos por el usuario, procesados e indexados con embeddings. Ver `src/ingestion/` y `src/pipeline.py`. |
| **Preprocesar y transformar los datos** | Limpieza de texto + segmentación (*chunking*) con solapamiento. Ver `src/preprocessing/`. |
| **Aplicar ≥ 1 técnica de ML** | **Clustering con KMeans** sobre los embeddings de los documentos (selección de *k* por silueta, etiquetado por TF-IDF, proyección PCA). Ver `src/ml/clustering.py`. |
| **Dashboard interactivo (Streamlit)** | App multipágina: Ingesta, Chatbot RAG, Búsqueda Semántica y Dashboard. Ver `streamlit_app.py` y `pages/`. |
| **Código en GitHub con README** | Este repositorio. |

### Requisitos específicos del Grupo 7

| Requisito | Cómo se cumple |
|-----------|----------------|
| **Carga y procesamiento de PDFs** | `src/ingestion/pdf_loader.py` (pypdf + OCR fallback con pytesseract/pymupdf para PDFs escaneados) + pipeline en `src/pipeline.py`. |
| **Indexación con ChromaDB** | `src/indexing/vector_store.py` (cliente persistente + embeddings). |
| **Búsqueda semántica** | Página *Búsqueda Semántica* + `vector_store.search()` (distancia coseno). |
| **Chatbot con RAG funcional** | `src/rag/rag_pipeline.py` + página *Chatbot RAG*. |
| **Dashboard con estadísticas de documentos** | Página *Dashboard* con métricas + clustering temático. |

---

## 🏗️ Arquitectura

```
                      ┌──────────────────────────────────────────────┐
   PDFs (usuario) ───▶│   PIPELINE DE DATOS (src/pipeline.py)        │
                      │  1. Ingesta   (ingestion/)                   │
                      │  2. Limpieza + chunking  (preprocessing/)    │
                      │  3. Embeddings + indexación  (indexing/)     │
                      └───────────────────┬──────────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   ChromaDB (vectores) │
                              └───────────┬───────────┘
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
          Búsqueda semántica       RAG (recuperar+LLM)     ML: Clustering KMeans
                  │                       │                       │
                  └───────────────────────┴───────────────────────┘
                                          ▼
                            DASHBOARD STREAMLIT (streamlit_app.py + pages/)
```

**Stack:** Python · ChromaDB · embeddings *all-MiniLM-L6-v2* (ONNX) · scikit-learn
· Groq/Llama 3 (open source) · Streamlit · Plotly.

---

## 🚀 Instalación y ejecución

### 1. Requisitos previos
- Python 3.10+ (probado en 3.11)
- Conexión a internet (la primera vez descarga el modelo de embeddings)
- **Tesseract OCR** (opcional, solo si necesitas indexar PDFs escaneados sin capa de texto)

#### Instalar Tesseract en Windows

1. Descarga el instalador desde **https://github.com/UB-Mannheim/tesseract/wiki** (versión `tesseract-ocr-w64-setup-5.x.x.exe`)
2. Durante la instalación, marca el idioma adicional **Spanish**
3. Verifica la instalación:
   ```powershell
   tesseract --version
   ```

> Si Tesseract no queda en el PATH, el sistema lo busca automáticamente en `C:\Program Files\Tesseract-OCR\`. También puedes sobrescribir la ruta en `.env`:
> ```env
> TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
> ```

### 2. Crear entorno virtual e instalar dependencias

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar el LLM (opcional pero recomendado)

Copia `.env.example` a `.env` y añade tu API key gratuita de Groq:

```powershell
Copy-Item .env.example .env
```

> Obtén una API key gratis en **https://console.groq.com/keys**.
> Si no configuras una API key, el chatbot funciona igual en **modo extractivo**
> (devuelve los fragmentos más relevantes), de modo que el sistema **siempre
> responde**.

#### ¿Qué es Groq y por qué es gratis?

**Groq** es una plataforma que ofrece acceso a modelos LLM open source (Llama) alojados en la nube, con velocidad ultrarrápida. Ofrecen un tier gratuito generoso (~30 solicitudes/minuto) porque:
- Usan modelos open source (no pagan licencias)
- Buscan captar usuarios para monetizar después
- Compiten con OpenAI ofreciendo alternativas gratis

**Modelos disponibles gratis en Groq:**

| Modelo | Parámetros | Velocidad | Mejor para | Estado |
|---|---|---|---|---|
| Llama 3.3 70B Versatile | 70B | ⚡⚡ Rápido | Calidad alta, respuestas precisas | ✅ Gratis |
| Llama 3.1 8B Instant | 8B | ⚡⚡⚡ Ultra rápido | Respuestas rápidas, tareas sencillas | ✅ **Actual en este proyecto** |
| Llama 3.1 70B Versatile | 70B | ⚡⚡ Rápido | Mejor calidad, más inteligente | ✅ Gratis |
| Llama 3.2 90B Vision | 90B | ⚡ Un poco lento | Análisis de imágenes + texto | ✅ Gratis |
| Mixtral 8x7B | Mezcla | ⚡⚡ Rápido | Tareas variadas | ✅ Gratis |
| Gemma 2 9B | 9B | ⚡⚡⚡ Ultra rápido | Alternativa ligera | ✅ Gratis |

**El modelo activo se configura en `.env`** con la variable `GROQ_MODEL`.

**Para cambiar de modelo,** edita `.env`:
```env
GROQ_MODEL=llama-3.1-8b-instant    # Actual — rápido y con buen rate limit
```

O si no tienes API key, el sistema funciona en **modo extractivo** — sin LLM, solo muestra los fragmentos más relevantes directamente.

### 4. Lanzar el dashboard

```powershell
streamlit run streamlit_app.py
```

Se abrirá en `http://localhost:8501`.

### 5. Alternativas: Otros LLMs

Tu sistema soporta tres modos de LLM:

#### **Opción 1: Groq (recomendado)** ✅

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_xxxxx...
GROQ_MODEL=llama-3.1-8b-instant
```
- ✅ Gratis (tier generoso)
- ✅ Muy rápido (~1 segundo por respuesta)
- ✅ Sin instalación local
- ❌ Requiere internet

#### **Opción 2: Ollama (100% offline)**

Instala [Ollama](https://ollama.ai) y corre un modelo localmente:

```powershell
# Descarga e instala Ollama (una sola vez)
# Luego corre un modelo:
ollama run llama3.1

# En otra terminal, inicia tu proyecto
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Configura en `.env`:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://localhost:11434
```

- ✅ 100% offline (sin internet)
- ✅ Sin API keys
- ❌ Requiere 8 GB RAM + GPU opcional
- ❌ Más lento que Groq

#### **Opción 3: Modo extractivo (sin LLM)**

```env
LLM_PROVIDER=none
# (o simplemente no configures GROQ_API_KEY)
```

- ✅ No necesita nada
- ✅ Rápido
- ❌ Solo muestra fragmentos, sin generación de respuestas

---

## 🧰 Uso por línea de comandos (pipeline)

```powershell
# Indexar todos los PDFs de una carpeta
python scripts/run_pipeline.py --pdf-dir data/raw

# Indexar un único PDF
python scripts/run_pipeline.py --pdf ruta/archivo.pdf

# Reconstruir el índice desde cero
python scripts/run_pipeline.py --pdf-dir data/raw --reset
```

---

## 📁 Estructura del proyecto

```
Parcial 2/
├── streamlit_app.py          # Dashboard (inicio)
├── app_helpers.py            # Utilidades compartidas de Streamlit
├── pages/                    # Páginas del dashboard
│   ├── 1_📥_Ingesta_de_Documentos.py
│   ├── 2_💬_Chatbot_RAG.py
│   ├── 3_🔎_Búsqueda_Semántica.py
│   └── 4_📊_Dashboard.py
├── src/
│   ├── config.py             # Configuración (variables de entorno)
│   ├── pipeline.py           # Orquestador del pipeline
│   ├── ingestion/            # Carga de PDFs
│   ├── preprocessing/        # Limpieza + chunking
│   ├── indexing/             # ChromaDB + búsqueda semántica
│   ├── ml/                   # Clustering KMeans (ML)
│   └── rag/                  # LLM + pipeline RAG
├── scripts/
│   └── run_pipeline.py       # CLI del pipeline
├── data/                     # raw/ , processed/ , chroma/ (generados)
├── requirements.txt
├── .env.example
├── PLAN.md · AGENT.md · CLAUDE.md   # Contexto para asistentes de IA
└── README.md
```

---

## 🧪 ¿Cómo funciona el RAG?

1. **Indexación:** cada PDF se divide en fragmentos; cada fragmento se convierte
   en un vector (embedding) y se guarda en ChromaDB.
2. **Recuperación:** ante una pregunta, se busca por similitud semántica los *k*
   fragmentos más parecidos.
3. **Generación:** esos fragmentos se pasan como contexto a un LLM open source
   (Llama 3 vía Groq), que redacta la respuesta citando las fuentes.
4. **Respaldo:** si no hay LLM disponible, se devuelve una respuesta extractiva
   con los fragmentos más relevantes.

---

## 👥 Integrantes — Grupo 7

| Nombre | Cédula |
|--------|--------|
| Kelvin He | 8-999-1950 |
| Roy Barrera | 8-1022-2121 |
| Angélica Rodríguez | 2-751-41 |
| Alex Jiménez | 8-987-473 |
| Jack García | 9-958-2038 |

**Profesor:** Reinel Aguirre · **Salón:** 1GS241

---

## 📄 Licencia

Proyecto académico desarrollado con fines educativos para la Universidad
Tecnológica de Panamá.
