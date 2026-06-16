# Estructura y funcionamiento del proyecto
**Sistema de Gestión de Documentos Académicos con RAG — Grupo 7**

---

## Tabla de contenido

1. [¿Por qué ChromaDB?](#1-por-qué-chromadb)
2. [Qué hay dentro de la carpeta `data/chroma/`](#2-qué-hay-dentro-de-la-carpeta-datachroma)
3. [Mapa completo del proyecto](#3-mapa-completo-del-proyecto)
4. [Archivos raíz](#4-archivos-raíz)
5. [Carpeta `src/` — el motor del sistema](#5-carpeta-src--el-motor-del-sistema)
6. [Carpeta `pages/` — las páginas del dashboard](#6-carpeta-pages--las-páginas-del-dashboard)
7. [Carpeta `scripts/` — herramientas CLI](#7-carpeta-scripts--herramientas-cli)
8. [Carpeta `data/` — almacenamiento de datos](#8-carpeta-data--almacenamiento-de-datos)
9. [Flujo completo de datos](#9-flujo-completo-de-datos)

---

## 1. ¿Por qué ChromaDB?

### La diferencia entre una base de datos normal y una base de datos vectorial

Una base de datos **normal** (como MySQL o SQLite) guarda filas y columnas de texto o números, y busca por coincidencia exacta. Si buscas "redes neuronales" solo encuentra documentos que contengan exactamente esas palabras.

Una base de datos **vectorial** como ChromaDB guarda los documentos como **vectores** — listas de números que representan el *significado* del texto — y busca por *similitud de significado*. Si buscas "aprendizaje profundo", puede encontrar un documento que habla de "deep learning" aunque no use esas palabras exactas en español.

### ¿Qué es un vector (embedding)?

Imagina que cada fragmento de texto se convierte en una lista de 384 números (dimensiones). Textos con significado similar quedan cerca en ese espacio matemático. Por ejemplo:

```
"redes neuronales"  →  [0.12, -0.34, 0.87, ..., 0.05]  (384 números)
"deep learning"     →  [0.11, -0.33, 0.85, ..., 0.06]  (muy parecido ↑)
"fútbol"            →  [-0.9,  0.71, 0.02, ..., -0.8]  (muy diferente)
```

Ese proceso de convertir texto en números se llama **embedding**, y lo hace el modelo `paraphrase-multilingual-MiniLM-L12-v2` (corre vía ONNX con `fastembed`, sin necesidad de GPU).

### ¿Por qué ChromaDB específicamente?

| Característica | ChromaDB | MySQL/SQLite |
|---|---|---|
| Busca por significado | ✅ | ❌ |
| Fácil de usar en Python | ✅ | ✅ |
| Guarda en disco (sin servidor externo) | ✅ | ✅ |
| Incluye modelo de embeddings | ✅ | ❌ |
| Requiere GPU | ❌ (usa ONNX) | — |
| Ideal para RAG | ✅ | ❌ |

ChromaDB se eligió porque: (1) no requiere instalar un servidor separado, (2) guarda los datos en disco para no reindexar cada vez que se reinicia el programa, (3) la búsqueda semántica es exactamente lo que necesita un sistema RAG, y (4) se integra fácilmente con embeddings locales de alta calidad.

---

## 2. Qué hay dentro de la carpeta `data/chroma/`

Esta carpeta es donde ChromaDB guarda **todo** de forma persistente en disco. Tiene esta estructura:

```
data/chroma/
├── 5e310444-62c7-4499-ace6-411ad985bd99/   ← carpeta de índice vectorial
│   ├── data_level0.bin
│   ├── header.bin
│   ├── length.bin
│   └── link_lists.bin
└── chroma.sqlite3                           ← base de datos de metadatos
```

### La carpeta con código UUID (`5e310444-62c7-4499-ace6-411ad985bd99`)

Ese nombre raro es un **UUID** (Universally Unique Identifier) — un código de 36 caracteres generado automáticamente por ChromaDB para identificar la colección de vectores de forma única en el mundo. No significa nada semántico; es simplemente un identificador interno.

Dentro de esa carpeta están los archivos del **índice HNSW** (Hierarchical Navigable Small World) — el algoritmo que usa ChromaDB para buscar vectores similares de forma ultrarrápida:

| Archivo | Qué guarda |
|---|---|
| `header.bin` | Metadatos del índice: cuántos vectores hay, cuántas dimensiones tienen (384), parámetros del algoritmo HNSW. Es como la "portada" del índice. |
| `data_level0.bin` | Los **vectores en sí** — todos los números (embeddings) de todos los fragmentos indexados, en formato binario comprimido. Es el archivo más pesado. |
| `length.bin` | El tamaño (longitud) de cada entrada en el índice, para que el sistema sepa cuántos bytes leer por cada vector. |
| `link_lists.bin` | Las **conexiones del grafo HNSW** — cada vector tiene enlaces a sus vecinos más cercanos para que la búsqueda salte de vecino en vecino eficientemente sin comparar contra todos. |

> **Por qué `.bin`?** Son archivos binarios (no son texto legible). Guardan los datos en el formato nativo de la memoria de la computadora, lo que los hace mucho más rápidos de leer y escribir que si fueran texto.

### `chroma.sqlite3`

Es una base de datos **SQLite** (el formato de base de datos más popular del mundo, incluido en Python). ChromaDB la usa para guardar todo lo que no son vectores:

- El texto original de cada fragmento
- Los metadatos: título del PDF, autor, número de páginas, fecha de indexación, etc.
- Las IDs de cada fragmento (por ejemplo `abc123::chunk0`, `abc123::chunk1`)
- El nombre y configuración de la colección (`documentos_academicos`)
- La relación entre cada ID y su posición en los archivos `.bin`

En resumen: **`chroma.sqlite3` guarda el texto y los metadatos; los archivos `.bin` guardan los vectores matemáticos**.

---

## 3. Mapa completo del proyecto

```
Parcial 2/
│
├── streamlit_app.py          ← Entry point con navegación y estilos globales
├── app_helpers.py            ← Funciones compartidas entre páginas
├── CLAUDE.md                 ← Instrucciones para Claude Code
├── AGENT.md                  ← Contexto para asistentes de IA
├── PLAN.md                   ← Plan del proyecto y mapeo de requisitos
├── README.md                 ← Documentación principal (para GitHub)
├── requirements.txt          ← Lista de dependencias Python
├── runtime.txt               ← Versión de Python para despliegue
├── .env.example              ← Plantilla de configuración
├── .env                      ← Tu configuración real (no se sube a Git)
├── .gitignore                ← Archivos que Git debe ignorar
├── assets/
│   └── favicon.png           ← Icono de la aplicación
│
├── src/                      ← TODO el código del sistema
│   ├── __init__.py           ← Paquete raíz
│   ├── config.py             ← Configuración global
│   ├── pipeline.py           ← Orquestador principal
│   ├── chat_sessions.py      ← Gestión persistente de sesiones de chat
│   ├── ingestion/            ← Carga de PDFs
│   │   ├── __init__.py
│   │   └── pdf_loader.py     ← Extracción con pypdf + OCR fallback
│   ├── preprocessing/        ← Limpieza y segmentación
│   │   ├── __init__.py
│   │   └── text_processing.py
│   ├── indexing/             ← Base de datos vectorial
│   │   ├── __init__.py
│   │   └── vector_store.py   ← ChromaDB + búsqueda híbrida BM25
│   ├── ml/                   ← Machine Learning (clustering)
│   │   ├── __init__.py
│   │   └── clustering.py
│   └── rag/                  ← Chatbot (LLM + recuperación)
│       ├── __init__.py
│       ├── llm.py            ← Proveedores de LLM
│       └── rag_pipeline.py   ← Pipeline RAG completo
│
├── pages/                    ← Páginas del dashboard Streamlit
│   ├── inicio.py             ← Página de bienvenida, métricas y guía
│   ├── ingesta.py            ← Subir y eliminar PDFs
│   ├── chatbot.py            ← Chatbot con sesiones persistentes
│   ├── busqueda.py           ← Búsqueda semántica directa
│   └── dashboard.py          ← Estadísticas y clustering
│
├── scripts/                  ← Herramientas de línea de comandos
│   ├── __init__.py
│   └── run_pipeline.py       ← CLI del pipeline
│
├── docs/                     ← Documentación del proyecto
│   └── ESTRUCTURA_DEL_PROYECTO.md   ← (este archivo)
│
├── data/                     ← Datos generados (no se suben a Git)
│   ├── raw/                  ← PDFs subidos por el usuario
│   ├── processed/            ← Carpeta reservada (uso futuro)
│   ├── chroma/               ← Base de datos vectorial persistente
│   └── sessions.json         ← Sesiones de chat persistentes
│
└── tests/                    ← Pruebas automáticas
    ├── test_smoke.py
    └── test_streamlit_pages.py
```

---

## 4. Archivos raíz

### `streamlit_app.py` — Punto de entrada principal

Es el archivo que se ejecuta cuando corres `streamlit run streamlit_app.py`. Ahora no es una página de inicio con métricas, sino el **orquestador de navegación** del dashboard.

**Qué hace:**
- Configura `st.set_page_config` con el favicon de `assets/favicon.png`
- Inyecta CSS global (tema oscuro, fuentes Inter, estilos de métricas, botones, chat, scrollbars, etc.)
- Define la navegación con `st.navigation()` apuntando a las 5 páginas en `pages/`
- Ejecuta `pg.run()` para iniciar la aplicación

> Las 5 páginas aparecen automáticamente en el menú lateral de Streamlit: Inicio, Ingesta, Chatbot, Búsqueda y Dashboard.

---

### `app_helpers.py` — Funciones compartidas

Evita repetir el mismo código en las 5 páginas del dashboard. Todas las páginas importan este archivo.

**Funciones:**

| Función | Qué hace |
|---|---|
| `sidebar_status()` | Muestra en la barra lateral: qué LLM está activo y cuántos fragmentos hay indexados |
| `cached_statistics(version)` | Pide al pipeline las estadísticas de documentos (con caché para no recalcular cada vez) |
| `cached_clustering(version, n_clusters)` | Ejecuta el clustering KMeans (con caché para no reentrenar en cada clic) |
| `data_version()` | Devuelve el número de fragmentos indexados — se usa para invalidar la caché cuando se suben nuevos PDFs |

> **¿Por qué `version` en las funciones cacheadas?** Streamlit cachea los resultados de las funciones. Si el número de fragmentos cambia (se subió un PDF nuevo), el parámetro `version` cambia y fuerza que se recalcule.

---

### `requirements.txt` — Dependencias

Lista todos los paquetes Python que necesita el proyecto:

```
streamlit>=1.49.0      → el framework del dashboard web
chromadb>=0.5.5,<0.6.0 → base de datos vectorial
protobuf>=3.20.0,<4.0.0 → dependencia de ChromaDB (compatibilidad)
onnxruntime>=1.18.0    → motor para correr modelos de embeddings sin GPU
fastembed>=0.4.0       → embeddings multilingües vía ONNX
pypdf>=5.1.0           → leer y extraer texto de archivos PDF
pymupdf>=1.24.0        → renderizado de páginas para OCR
pytesseract>=0.3.13    → OCR fallback (texto de imágenes escaneadas)
pillow>=10.0.0         → procesamiento de imágenes para OCR
groq>=0.13.0           → API para usar modelos LLM open source (Llama 3)
requests>=2.32.0       → cliente HTTP (para conectar con Ollama si se usa)
rank-bm25>=0.2.2       → búsqueda híbrida por palabras clave exactas
scikit-learn>=1.5.0    → KMeans, PCA, TF-IDF, coeficiente de silueta
pandas>=2.2.0          → manejo de tablas de datos
numpy>=1.26.0          → operaciones matemáticas con vectores
plotly>=5.24.0         → gráficas interactivas en el dashboard
python-dotenv>=1.0.1   → leer el archivo .env con las configuraciones
```

---

### `.env` / `.env.example`

`.env` es el archivo donde guardas tus claves privadas y configuraciones. **Nunca se sube a GitHub** (está en `.gitignore`). `.env.example` es la plantilla vacía que sí se puede compartir.

Variables configurables:

```
LLM_PROVIDER=groq                → puede ser "groq", "ollama" o "none"
GROQ_API_KEY=                    → tu clave de API de Groq (gratis en console.groq.com)
GROQ_MODEL=llama-3.1-8b-instant  → modelo a usar vía Groq
OLLAMA_MODEL=llama3.1            → modelo a usar vía Ollama (local)
OLLAMA_HOST=http://localhost:11434 → URL del servidor Ollama
COLLECTION_NAME=documentos_academicos  → nombre de la colección en ChromaDB
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=900                   → tamaño de cada fragmento en caracteres
CHUNK_OVERLAP=150                → solapamiento entre fragmentos
TOP_K=5                          → cuántos fragmentos recuperar por búsqueda
LLM_TEMPERATURE=0.2              → creatividad del LLM (0 = conservador, 1 = creativo)
```

---

## 5. Carpeta `src/` — el motor del sistema

### `src/config.py` — Configuración global

Es el primer archivo que se carga. Lee el `.env` y define constantes que usa todo el sistema.

**Qué define:**

```python
BASE_DIR         → ruta raíz del proyecto (calculada automáticamente)
DATA_DIR         → data/
RAW_DIR          → data/raw/      (donde se guardan los PDFs)
PROCESSED_DIR    → data/processed/ (reservada)
CHROMA_DIR       → data/chroma/   (donde vive la base de datos)

COLLECTION_NAME    → "documentos_academicos" (nombre de la colección en Chroma)
EMBEDDING_MODEL    → "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE       → 900 caracteres por fragmento
CHUNK_OVERLAP    → 150 caracteres de solapamiento

TOP_K            → 5 fragmentos a recuperar por búsqueda
MAX_CHUNKS_PER_DOC → 2 (máximo de chunks del mismo documento en el ranking)
MIN_SCORE        → 0.10 (score mínimo para incluir un fragmento)

LLM_PROVIDER     → "groq" / "ollama" / "none"
GROQ_API_KEY     → tu API key
GROQ_MODEL       → "llama-3.1-8b-instant"
OLLAMA_MODEL     → "llama3.1"
OLLAMA_HOST      → "http://localhost:11434"
LLM_TEMPERATURE  → 0.2
```

También crea automáticamente las carpetas `raw/`, `processed/` y `chroma/` si no existen, para que el sistema nunca falle por una carpeta que falta.

**Función `llm_status()`:**
Devuelve una descripción del proveedor de LLM activo (por ejemplo: "Groq — llama-3.1-8b-instant" o "Sin LLM — respuesta extractiva").

---

### `src/pipeline.py` — Orquestador

Es el "director de orquesta" del sistema. Conecta todos los módulos y ofrece las funciones que usa tanto el dashboard como el CLI.

**Funciones principales:**

#### `ingest_pdf_file(path, extra_metadata=None)` → Indexa un PDF

Paso a paso:
1. Llama a `pdf_loader.extract_pdf(path)` para extraer el texto y metadatos del PDF
2. Llama a `clean_text()` para limpiar el texto (quitar caracteres raros, normalizar espacios)
3. Llama a `chunk_text()` para dividir el texto en fragmentos de ~900 caracteres
4. Genera un `doc_id` único (hash MD5 del título + nombre de archivo)
5. Llama a `vector_store.add_document()` para indexar los fragmentos en ChromaDB
6. Añade `char_count` (cantidad de caracteres del texto) a los metadatos
7. Devuelve un resumen: `{"ok": True, "title": "...", "chunks": 42, ...}`

> El parámetro `extra_metadata` permite añadir metadatos adicionales (por ejemplo, una etiqueta o categoría) al momento de indexar.

#### `ingest_pdf_directory(directory)` → Indexa una carpeta entera

Recorre todos los `.pdf` de la carpeta y llama a `ingest_pdf_file` para cada uno.

#### `get_statistics()` → Estadísticas para el dashboard

Consulta ChromaDB para saber cuántos documentos, fragmentos y páginas hay, y los agrupa por fuente. Ahora incluye el campo `published` en la información de documentos.

---

### `src/ingestion/pdf_loader.py` — Lector de PDFs

Usa la librería `pypdf` (`PdfReader`) como método principal para extraer texto de archivos PDF.

**Función `extract_pdf(path)`:**
- Abre el PDF con `pypdf.PdfReader`
- Extrae el texto de cada página (si una página está corrupta, la salta sin abortar)
- Si el texto es muy corto por página (< 50 caracteres, indicativo de PDF escaneado), activa el **fallback OCR**:
  - Usa `pymupdf` (fitz) para renderizar la página como imagen
  - Usa `pytesseract` para leer el texto de la imagen
  - Busca Tesseract en rutas típicas de Windows o en la variable `TESSERACT_CMD`
- Lee los metadatos del PDF (título, autor) si los tiene; si no, usa el nombre del archivo
- Devuelve: `{title, author, num_pages, text, filename, source}`

> `source` siempre vale `"local"` porque los PDFs los sube el usuario.

---

### `src/preprocessing/text_processing.py` — Limpieza y segmentación

#### `clean_text(text)` — Limpia el texto crudo

Los PDFs extraídos suelen tener caracteres raros, guiones de separación de palabras y saltos de línea innecesarios. Esta función los corrige:

1. Elimina caracteres nulos (`\x00`) y caracteres de control invisibles
2. Une palabras partidas por guion al final de línea: `"infor-\nmación"` → `"información"`
3. Normaliza saltos de línea: 2 o más saltos = párrafo nuevo; 1 salto = espacio
4. Colapsa espacios múltiples en uno solo

#### `chunk_text(text)` — Divide en fragmentos

Divide el texto limpio en fragmentos de ~900 caracteres con 150 caracteres de solapamiento.

**¿Por qué solapamiento?** Imagina un fragmento que termina a mitad de una idea. El fragmento siguiente empieza 150 caracteres antes del corte, para que esa idea aparezca completa en al menos uno de los fragmentos. Esto mejora la calidad de la búsqueda.

```
Texto original:  [====================================]
Fragmento 1:     [============]
Fragmento 2:          [============]   ← empieza 150 chars antes del fin del 1
Fragmento 3:               [============]
```

El corte respeta los límites de oraciones cuando es posible (no corta a mitad de una frase).

---

### `src/indexing/vector_store.py` — Base de datos vectorial (ChromaDB + híbrida)

Es la interfaz entre el sistema y ChromaDB. Todos los demás módulos usan este archivo para guardar o buscar datos. **Este módulo cambió radicalmente** y ahora implementa una búsqueda híbrida que combina semántica + keyword.

#### `get_collection()` — Obtiene la colección

Abre (o crea si no existe) la colección `"documentos_academicos"` en ChromaDB, configurada para usar distancia coseno.

#### `_FastEmbedFunction` — Embedding function personalizada

Ya no usa el embedding function default de ChromaDB. Ahora usa `fastembed` con el modelo `paraphrase-multilingual-MiniLM-L12-v2` a través de una clase interna `_FastEmbedFunction`. Esto permite embeddings multilingües de alta calidad corriendo localmente vía ONNX.

#### `add_document(doc_id, chunks, metadata)` — Indexa fragmentos

Para cada fragmento del texto:
- Genera un ID único: `doc_id::chunk0`, `doc_id::chunk1`, etc.
- Le añade metadatos: título, autor, número de páginas, fecha de indexación, índice del fragmento
- Usa `upsert` (insert + update): si el documento ya estaba indexado, lo actualiza en lugar de duplicarlo
- Invalida la caché BM25 (`_bm25_cache`) para que la próxima búsqueda reconstruya el índice keyword

#### `search(query, k=5)` — Búsqueda híbrida (semántica + BM25)

Ahora no es solo búsqueda semántica. Combina **tres técnicas** en una sola:

1. **Búsqueda semántica**: convierte la consulta en un vector con `fastembed` y busca los vectores más cercanos en ChromaDB usando distancia coseno.
2. **Búsqueda BM25** (`rank_bm25.BM25Okapi`): busca coincidencias exactas de palabras en el texto de los fragmentos. Es ideal para términos técnicos, nombres propios o códigos exactos.
3. **Fusión RRF** (Reciprocal Rank Fusion): combina los rankings de ambas búsquedas con la fórmula `1 / (rank + 60)`, dando una puntuación unificada a cada fragmento.

Después de la fusión, aplica dos filtros adicionales:
- **Límite por documento** (`MAX_CHUNKS_PER_DOC`): asegura que no haya más de 2 chunks del mismo PDF en los resultados, promoviendo diversidad de fuentes.
- **Filtro de score mínimo** (`MIN_SCORE`): descarta fragmentos con score inferior a 0.10 para evitar resultados irrelevantes.

#### `search_filtered(query, k, doc_ids)` — Búsqueda restringida

Igual que `search()`, pero restringe los resultados a un conjunto de documentos específicos (`doc_ids`). Además, implementa **garantía de cobertura**: siempre devuelve al menos 1 chunk de cada documento seleccionado, para que ningún documento elegido por el usuario quede excluido por el ranking.

#### `delete_document(doc_id)` — Elimina un documento

Elimina todos los chunks de un documento del índice. Útil cuando el usuario quiere remover un PDF ya indexado.

#### `list_documents()` — Lista documentos únicos

Devuelve todos los documentos indexados con su conteo de chunks, para mostrar en el dashboard.

#### `get_all(include)` — Obtiene todos los datos

Devuelve todos los chunks, embeddings o metadatos. Se usa para estadísticas y para el módulo de ML (clustering).

#### `reset_collection()` — Borra todo

Elimina completamente la colección. Se usa cuando se quiere empezar desde cero.

#### `_sanitize_metadata(meta)` — Limpia los metadatos

ChromaDB solo acepta valores de tipo `str`, `int`, `float` o `bool` en los metadatos. Esta función convierte listas a texto separado por comas y elimina los valores `None` para evitar errores.

#### `_bm25_cache` — Caché del índice BM25

Para no reconstruir el índice BM25 en cada búsqueda, se guarda en caché junto con una "versión" (hash de los IDs de todos los documentos). Si la versión cambia (se indexó o eliminó un PDF), se reconstruye automáticamente.

---

### `src/ml/clustering.py` — Machine Learning (KMeans)

Esta es la técnica de ML del proyecto. Agrupa los documentos por similitud temática.

**Flujo completo:**

1. **`_document_vectors()`** — Obtiene los embeddings de todos los fragmentos desde ChromaDB y los *promedia* por documento (si un documento tiene 40 fragmentos, su vector es el promedio de los 40). Resultado: una matriz donde cada fila es un documento.

2. **`_best_k(X, k_max)`** — Prueba distintos valores de k (de 2 a `k_max`) y elige el que da el mayor **coeficiente de silueta** (métrica que mide qué tan bien separados están los clusters entre -1 y 1).

3. **`KMeans`** — Entrena el algoritmo con el k elegido. Cada documento queda asignado a un cluster.

4. **`PCA`** — Reduce los vectores de 384 dimensiones a 2 dimensiones para poder graficarlos en una pantalla (es imposible graficar 384 ejes). Se pierde algo de información pero se mantiene la estructura general.

5. **`_top_terms()`** — Usa TF-IDF para encontrar las palabras más representativas de cada cluster y así darle una "etiqueta temática" (por ejemplo: "neural, network, training" para el cluster de ML).

**Función principal `cluster_documents()`** devuelve todo listo para el dashboard: k, coeficiente de silueta, posiciones 2D de cada documento, etiquetas de los clusters.

---

### `src/rag/llm.py` — Proveedor de LLM

Abstracción que permite cambiar de proveedor de lenguaje sin tocar el resto del código.

**Función `generate(prompt, system, history=[])`:**
- Ahora acepta un parámetro `history` con los turnos previos de la conversación, permitiendo diálogos **multi-turno**.
- Si `LLM_PROVIDER=groq` y hay `GROQ_API_KEY` → llama a la API de Groq (Llama 3 en la nube)
- Si `LLM_PROVIDER=ollama` → llama a un modelo local corriendo en tu máquina
- Si ninguno está disponible → devuelve `None` (el sistema usa respuesta extractiva)

**Modo extractivo (fallback):** Si no hay LLM, el sistema no falla. En lugar de generar texto con IA, simplemente muestra los 3 fragmentos más relevantes que encontró. Siempre responde algo útil.

---

### `src/rag/rag_pipeline.py` — Pipeline RAG completo (multi-turno)

Combina la búsqueda híbrida con la generación de respuesta, ahora con soporte para **sesiones persistentes** y **contexto conversacional**.

**Función `answer(question, k=None, doc_ids=None, history=None)`:**

1. **`_expand_query(question, history)`** — Si la pregunta es corta (≤ 8 palabras) y hay un turno anterior en el historial, expande la query automáticamente combinándola con la pregunta previa. Esto mejora el retrieval para preguntas de seguimiento como "¿Y qué más?" o "¿Cómo funciona?".

2. **`vector_store.search()`** o **`search_filtered()`** — Busca los fragmentos más relevantes:
   - Si `doc_ids` es `None` → busca en todos los documentos (`search()`)
   - Si `doc_ids` tiene IDs → busca solo en esos documentos (`search_filtered()`)

3. **`_build_context(hits, prev_answer)`** — Construye el contexto concatenando los fragmentos. Si hay una respuesta previa (`prev_answer`), la antepone como contexto adicional para que el LLM tenga continuidad en la conversación.

4. Construye el **prompt** completo con el contexto y la pregunta:
   ```
   CONTEXTO:
   [Fragmento 1 | Fuente: Nombre del PDF]
   texto del fragmento...

   [Fragmento 2 | Fuente: Otro PDF]
   texto del fragmento...

   PREGUNTA: ¿Cuál es la definición de X?

   RESPUESTA (basada solo en el contexto anterior):
   ```

5. Envía el prompt al LLM con el historial de mensajes → recibe la respuesta generada
6. Si el LLM falla → usa `_extractive_answer(hits)` (muestra los fragmentos directamente)
7. Devuelve: `{answer, sources, provider}`

**`SYSTEM_PROMPT`:** Le dice al LLM que responda **solo** con base en el contexto proporcionado, en **español**, y que **no mencione números de fragmento** (como "fragmento 1" o "chunk 2") en la respuesta. Si no sabe la respuesta, debe decirlo en lugar de inventar.

---

### `src/chat_sessions.py` — Gestión de sesiones de chat (NUEVO)

Persiste las sesiones de chat en `data/sessions.json` para que sobrevivan a recargas de página y reinicios del servidor.

**Funciones principales:**

| Función | Qué hace |
|---|---|
| `create_session(name)` | Crea una nueva sesión con nombre, ID único y fecha de creación |
| `list_sessions()` | Lista todas las sesiones existentes |
| `get_session(id)` | Obtiene una sesión por su ID |
| `rename_session(id, name)` | Cambia el nombre de una sesión |
| `set_session_docs(id, doc_ids)` | Asocia documentos específicos a una sesión (contexto restringido) |
| `delete_session(id)` | Elimina una sesión completamente |
| `add_message(id, role, content, sources)` | Añade un mensaje (usuario o asistente) al historial de la sesión |
| `clear_messages(id)` | Borra el historial de mensajes de una sesión |
| `remove_doc_from_all_sessions(doc_id)` | Elimina un documento de todas las sesiones que lo tengan seleccionado (útil cuando se borra un PDF del índice) |

**Estructura de una sesión:**
```json
{
  "id": "uuid-1234",
  "name": "Consulta sobre redes",
  "doc_ids": ["doc_abc", "doc_def"],
  "messages": [
    {"role": "user", "content": "¿Qué es una red neuronal?", "sources": []},
    {"role": "assistant", "content": "Es un modelo computacional...", "sources": [{"title": "..."}]}
  ],
  "created_at": "2026-06-16T10:00:00"
}
```

---

## 6. Carpeta `pages/` — las páginas del dashboard

Streamlit detecta automáticamente los archivos en esta carpeta y los convierte en páginas del menú lateral. La navegación se define en `streamlit_app.py` con `st.navigation()`.

---

### `pages/inicio.py` — Página de bienvenida (NUEVO)

Es la página principal que ve el usuario al abrir la aplicación. Muestra un banner con gradiente, tags tecnológicas y métricas del sistema.

**Secciones:**
1. **Banner de bienvenida** — título del proyecto con gradiente visual
2. **Tags tecnológicas** — chips visuales que muestran: ChromaDB, Sentence Transformers, KMeans ML, Llama 3 via Groq
3. **Métricas rápidas** — tarjetas con: Documentos indexados, Fragmentos totales, Páginas escaneadas, Fuentes únicas
4. **Guía de uso visual** — paso a paso en 4 pasos:
   - Paso 1: Sube tus PDFs en la página de Ingesta
   - Paso 2: Espera que el sistema los procese y indexe
   - Paso 3: Ve al Chatbot para hacer preguntas
   - Paso 4: Explora el Dashboard para estadísticas y clustering
5. **Advertencias contextuales** —
   - Si no hay API key de Groq configurada, muestra un aviso con enlace a console.groq.com
   - Si no hay documentos indexados, muestra una guía para ir a la página de Ingesta

---

### `pages/ingesta.py` — Subir y eliminar PDFs

Permite al usuario gestionar los documentos del sistema.

**Flujo de subida:**
1. El usuario arrastra/selecciona uno o varios PDFs
2. Hace clic en "Procesar e indexar"
3. El sistema guarda el PDF en `data/raw/`, lo procesa con el pipeline y lo indexa en ChromaDB
4. Se muestra un resumen: cuántos fragmentos generó cada PDF
5. Se limpian las cachés de Streamlit (`st.cache_data.clear()`) para que el dashboard y el clustering se actualicen

**Indexar carpeta:**
- Botón para indexar todos los PDFs que ya estén en `data/raw/` sin tener que subirlos otra vez.

**Eliminar documentos (NUEVO):**
- Lista los documentos ya indexados con un botón de eliminar junto a cada uno.
- Pide confirmación antes de borrar (para evitar borrados accidentales).
- Al eliminar, también llama a `remove_doc_from_all_sessions()` para desvincular el documento de las sesiones de chat.
- Limpia cachés de Streamlit tras la eliminación.

---

### `pages/chatbot.py` — Chatbot con sesiones persistentes (CAMBIÓ MUCHO)

Interfaz de chat para hacer preguntas en lenguaje natural sobre los documentos. Ahora soporta **sesiones persistentes**, **selección de documentos** y **chat multi-turno**.

**Sidebar de sesiones:**
- **Crear sesión**: botón para crear una nueva conversación con nombre personalizado
- **Seleccionar sesión**: dropdown para elegir entre sesiones existentes
- **Renombrar sesión**: cambia el nombre de la sesión activa
- **Eliminar sesión**: borra la sesión y su historial

**Selección de documentos:**
- Multiselect para elegir qué documentos usar como contexto en la sesión actual.
- Si no se selecciona ninguno, la búsqueda se hace sobre todos los documentos.
- Los documentos seleccionados se guardan en la sesión y persisten entre recargas.

**Chat multi-turno:**
- El historial de mensajes se guarda en `data/sessions.json` y **sobrevive a recargas de página**.
- Cada pregunta incluye el historial previo como contexto para el LLM.
- **Query expansion automática**: para preguntas de seguimiento cortas (≤ 8 palabras), el sistema expande automáticamente la query con la pregunta anterior para mejorar la recuperación de información.

**Configuración:**
- Slider para controlar `top_k` (1-10): cuántos fragmentos recuperar por pregunta.

**Fuentes:**
- Cada respuesta incluye una sección expandible "Fuentes" que muestra qué documentos y fragmentos usó el sistema.

**Flujo de una conversación:**
1. El usuario escribe una pregunta en el cuadro de chat
2. Se llama a `rag_pipeline.answer(pregunta, k=top_k, doc_ids=seleccionados, history=historial)`
3. Se muestra la respuesta del LLM (o la extractiva si no hay LLM)
4. Se guarda el mensaje en `data/sessions.json` vía `chat_sessions.add_message()`
5. El historial se renderiza en la interfaz de chat

---

### `pages/busqueda.py` — Búsqueda semántica directa

Permite buscar fragmentos por significado sin pasar por el LLM.

**Diferencia con el chatbot:** el chatbot genera una respuesta nueva usando los fragmentos + IA. La búsqueda semántica simplemente muestra los fragmentos más similares tal como están en los documentos, con su score de relevancia.

**Nuevo:** Ahora usa la **búsqueda híbrida** (`vector_store.search()`), que combina embeddings semánticos + BM25 + RRF. Esto mejora los resultados para términos técnicos exactos y nombres propios.

Muestra una tabla resumen y luego cada fragmento expandible con su texto completo, metadatos y score de relevancia.

---

### `pages/dashboard.py` — Estadísticas y clustering

Visualiza el estado de la colección de documentos con gráficas interactivas.

**Secciones:**

1. **Métricas generales** — total de documentos, fragmentos, páginas y promedio de fragmentos por documento

2. **Distribución por fuente** — gráfica de torta mostrando cuántos documentos vienen de cada fuente (`local` si los subió el usuario)

3. **Fragmentos por documento** — gráfica de barras horizontales mostrando cuáles documentos tienen más fragmentos

4. **Tabla de documentos** — lista de todos los PDFs indexados con título, fuente, páginas y autor

5. **Clustering KMeans (ML)** — agrupa los documentos por similitud temática y muestra:
   - El mapa 2D de documentos (proyección PCA, cada punto es un documento coloreado por cluster)
   - Las palabras clave de cada cluster (TF-IDF)
   - El coeficiente de silueta (calidad del agrupamiento)
   - Toggle para dejar que el sistema elija k automáticamente o elegirlo manualmente

> Tras ingestar nuevos PDFs, el dashboard se actualiza automáticamente porque `pages/ingesta.py` limpia las cachés de Streamlit.

---

## 7. Carpeta `scripts/` — herramientas CLI

### `run_pipeline.py` — Ejecutar el pipeline desde la terminal

Permite indexar PDFs directamente desde la consola sin abrir el dashboard. Útil para procesar muchos PDFs de una vez o para automatizar tareas.

**Uso:**
```powershell
# Indexar todos los PDFs de una carpeta
python scripts/run_pipeline.py --pdf-dir data/raw

# Indexar un único PDF
python scripts/run_pipeline.py --pdf ruta/archivo.pdf

# Borrar el índice y reindexar desde cero
python scripts/run_pipeline.py --pdf-dir data/raw --reset
```

---

## 8. Carpeta `data/` — almacenamiento de datos

```
data/
├── raw/          ← PDFs que sube el usuario (se indexan desde aquí)
├── processed/    ← reservada para uso futuro (.gitkeep la mantiene en Git)
├── chroma/       ← base de datos vectorial persistente (ver sección 2)
└── sessions.json ← sesiones de chat persistentes (no se sube a Git)
```

> La carpeta `data/` completa está en `.gitignore` — no se sube a GitHub. Cada persona que clone el proyecto empieza con la base de datos vacía y sube sus propios PDFs. La única excepción es `data/processed/.gitkeep`, que mantiene la estructura de carpetas en Git.

---

## 9. Flujo completo de datos

Este es el recorrido que hace un PDF desde que el usuario lo sube hasta que puede hacer preguntas sobre él:

```
Usuario sube un PDF
        │
        ▼
pages/ingesta.py
  → guarda el archivo en data/raw/
  → llama a pipeline.ingest_pdf_file()
        │
        ▼
src/ingestion/pdf_loader.py
  → extract_pdf(): abre el PDF con pypdf.PdfReader
  → extrae texto y metadatos
  → si el texto es muy corto (< 50 chars/página):
       renderiza la página con pymupdf (fitz)
       aplica OCR con pytesseract
  → devuelve: {title, author, num_pages, text, filename, source}
        │
        ▼
src/preprocessing/text_processing.py
  → clean_text(): limpia caracteres raros, normaliza espacios
  → chunk_text(): divide en fragmentos de ~900 chars con solapamiento
        │
        ▼
src/indexing/vector_store.py
  → add_document(): envía los fragmentos a ChromaDB (upsert)
  → invalida la caché BM25 para reconstruir el índice keyword
        │
        ▼
ChromaDB (internamente) + fastembed
  → paraphrase-multilingual-MiniLM-L12-v2 convierte cada fragmento
    en un vector de 384 números vía ONNX
  → guarda el vector en data_level0.bin
  → guarda el texto y metadatos en chroma.sqlite3
  → actualiza el grafo HNSW en link_lists.bin
        │
        ▼
Usuario hace una pregunta en el chatbot
        │
        ▼
src/rag/rag_pipeline.py → answer(question, history, doc_ids)
        │
        ├─→ _expand_query(question, history)
        │     → si la pregunta es corta (≤ 8 palabras) y hay historial,
        │        expande la query con la pregunta anterior
        │
        ├─→ src/indexing/vector_store.py → search() o search_filtered()
        │     → Búsqueda semántica: convierte la pregunta en vector
        │       con fastembed y busca los vectores más cercanos (coseno)
        │     → Búsqueda BM25: busca palabras exactas en los fragmentos
        │     → Fusión RRF (k=60): combina ambos rankings
        │     → Límite MAX_CHUNKS_PER_DOC por documento
        │     → Filtro MIN_SCORE ≥ 0.10
        │     → devuelve fragmentos + scores unificados
        │
        ├─→ _build_context(hits, prev_answer)
        │     → si hay respuesta previa, la antepone como contexto
        │       adicional para continuidad conversacional
        │
        ├─→ construye el prompt con el contexto
        │
        └─→ src/rag/llm.py → generate(prompt, system, history)
              → Groq API (Llama 3) genera la respuesta con historial
              → si no hay LLM → respuesta extractiva
                      │
                      ▼
              src/chat_sessions.py → add_message()
                → guarda la pregunta y la respuesta en data/sessions.json
                      │
                      ▼
              Se muestra la respuesta al usuario
              con las fuentes citadas
```

---

*Documentación generada para el Proyecto Integrador — Segundo Parcial · UTP · Grupo 7*
