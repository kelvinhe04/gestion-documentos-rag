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

Ese proceso de convertir texto en números se llama **embedding**, y lo hace el modelo `all-MiniLM-L6-v2` (incluido en ChromaDB, corre sin GPU gracias a ONNX).

### ¿Por qué ChromaDB específicamente?

| Característica | ChromaDB | MySQL/SQLite |
|---|---|---|
| Busca por significado | ✅ | ❌ |
| Fácil de usar en Python | ✅ | ✅ |
| Guarda en disco (sin servidor externo) | ✅ | ✅ |
| Incluye modelo de embeddings | ✅ | ❌ |
| Requiere GPU | ❌ (usa ONNX) | — |
| Ideal para RAG | ✅ | ❌ |

ChromaDB se eligió porque: (1) no requiere instalar un servidor separado, (2) incluye los embeddings listos para usar, (3) guarda los datos en disco para no reindexar cada vez que se reinicia el programa, y (4) la búsqueda semántica es exactamente lo que necesita un sistema RAG.

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
├── streamlit_app.py          ← Página de inicio del dashboard
├── app_helpers.py            ← Funciones compartidas entre páginas
├── CLAUDE.md                 ← Instrucciones para Claude Code
├── AGENT.md                  ← Contexto para asistentes de IA
├── PLAN.md                   ← Plan del proyecto y mapeo de requisitos
├── README.md                 ← Documentación principal (para GitHub)
├── requirements.txt          ← Lista de dependencias Python
├── .env.example              ← Plantilla de configuración
├── .env                      ← Tu configuración real (no se sube a Git)
├── .gitignore                ← Archivos que Git debe ignorar
│
├── src/                      ← TODO el código del sistema
│   ├── config.py             ← Configuración global
│   ├── pipeline.py           ← Orquestador principal
│   ├── ingestion/            ← Carga de PDFs
│   ├── preprocessing/        ← Limpieza y segmentación
│   ├── indexing/             ← Base de datos vectorial
│   ├── ml/                   ← Machine Learning (clustering)
│   └── rag/                  ← Chatbot (LLM + recuperación)
│
├── pages/                    ← Páginas del dashboard Streamlit
│   ├── 1_📥_Ingesta_de_Documentos.py
│   ├── 2_💬_Chatbot_RAG.py
│   ├── 3_🔎_Búsqueda_Semántica.py
│   └── 4_📊_Dashboard.py
│
├── scripts/                  ← Herramientas de línea de comandos
│   └── run_pipeline.py
│
├── docs/                     ← Documentación del proyecto
│   └── ESTRUCTURA_DEL_PROYECTO.md   ← (este archivo)
│
├── data/                     ← Datos generados (no se suben a Git)
│   ├── raw/                  ← PDFs subidos por el usuario
│   ├── processed/            ← Carpeta reservada (uso futuro)
│   └── chroma/               ← Base de datos vectorial persistente
│
└── tests/                    ← Pruebas automáticas
    ├── test_smoke.py
    └── test_streamlit_pages.py
```

---

## 4. Archivos raíz

### `streamlit_app.py` — Página de inicio

Es el punto de entrada del dashboard. Cuando corres `streamlit run streamlit_app.py`, este archivo es el primero que se ejecuta.

**Qué hace:**
- Configura el título y el ícono de la aplicación web
- Muestra las métricas rápidas (cuántos documentos, fragmentos y páginas hay indexados)
- Muestra la guía de uso ("¿Cómo usar el sistema?")
- Muestra el estado del LLM (si hay API key de Groq configurada o no)
- Si no hay documentos indexados, muestra una advertencia guiando al usuario

---

### `app_helpers.py` — Funciones compartidas

Evita repetir el mismo código en las 4 páginas del dashboard. Todas las páginas importan este archivo.

**Funciones:**

| Función | Qué hace |
|---|---|
| `page_config(title, icon)` | Pone el título y el ícono en la pestaña del navegador |
| `sidebar_status()` | Muestra en la barra lateral: qué LLM está activo y cuántos fragmentos hay indexados |
| `cached_statistics(version)` | Pide al pipeline las estadísticas de documentos (con caché para no recalcular cada vez) |
| `cached_clustering(version, n_clusters)` | Ejecuta el clustering KMeans (con caché para no reentrenar en cada clic) |
| `data_version()` | Devuelve el número de fragmentos indexados — se usa para invalidar la caché cuando se suben nuevos PDFs |

> **¿Por qué `version` en las funciones cacheadas?** Streamlit cachea los resultados de las funciones. Si el número de fragmentos cambia (se subió un PDF nuevo), el parámetro `version` cambia y fuerza que se recalcule.

---

### `requirements.txt` — Dependencias

Lista todos los paquetes Python que necesita el proyecto:

```
streamlit    → el framework del dashboard web
chromadb     → base de datos vectorial
onnxruntime  → motor para correr el modelo de embeddings sin GPU
pypdf        → leer y extraer texto de archivos PDF
groq         → API para usar modelos LLM open source (Llama 3)
requests     → cliente HTTP (para conectar con Ollama si se usa)
scikit-learn → KMeans, PCA, TF-IDF, coeficiente de silueta
pandas       → manejo de tablas de datos
numpy        → operaciones matemáticas con vectores
plotly       → gráficas interactivas en el dashboard
python-dotenv → leer el archivo .env con las configuraciones
```

---

### `.env` / `.env.example`

`.env` es el archivo donde guardas tus claves privadas y configuraciones. **Nunca se sube a GitHub** (está en `.gitignore`). `.env.example` es la plantilla vacía que sí se puede compartir.

Variables configurables:

```
GROQ_API_KEY=        → tu clave de API de Groq (gratis en console.groq.com)
LLM_PROVIDER=groq    → puede ser "groq", "ollama" o "none"
GROQ_MODEL=llama-3.1-8b-instant  → modelo a usar
CHUNK_SIZE=900       → tamaño de cada fragmento en caracteres
CHUNK_OVERLAP=150    → solapamiento entre fragmentos
TOP_K=5              → cuántos fragmentos recuperar por búsqueda
```

---

## 5. Carpeta `src/` — el motor del sistema

### `src/config.py` — Configuración global

Es el primer archivo que se carga. Lee el `.env` y define constantes que usa todo el sistema.

**Qué define:**

```python
BASE_DIR       → ruta raíz del proyecto (calculada automáticamente)
DATA_DIR       → data/
RAW_DIR        → data/raw/    (donde se guardan los PDFs)
CHROMA_DIR     → data/chroma/ (donde vive la base de datos)

COLLECTION_NAME  → "documentos_academicos" (nombre de la colección en Chroma)
EMBEDDING_MODEL  → "all-MiniLM-L6-v2"

CHUNK_SIZE     → 900 caracteres por fragmento
CHUNK_OVERLAP  → 150 caracteres de solapamiento

TOP_K          → 5 fragmentos a recuperar por búsqueda

LLM_PROVIDER   → "groq" / "ollama" / "none"
GROQ_API_KEY   → tu API key
GROQ_MODEL     → "llama-3.1-8b-instant"
```

También crea automáticamente las carpetas `raw/`, `processed/` y `chroma/` si no existen, para que el sistema nunca falle por una carpeta que falta.

---

### `src/pipeline.py` — Orquestador

Es el "director de orquesta" del sistema. Conecta todos los módulos y ofrece las funciones que usa tanto el dashboard como el CLI.

**Funciones principales:**

#### `ingest_pdf_file(path)` → Indexa un PDF

Paso a paso:
1. Llama a `pdf_loader.extract_pdf(path)` para extraer el texto y metadatos del PDF
2. Llama a `clean_text()` para limpiar el texto (quitar caracteres raros, normalizar espacios)
3. Llama a `chunk_text()` para dividir el texto en fragmentos de ~900 caracteres
4. Genera un `doc_id` único (hash MD5 del título + nombre de archivo)
5. Llama a `vector_store.add_document()` para indexar los fragmentos en ChromaDB
6. Devuelve un resumen: `{"ok": True, "title": "...", "chunks": 42, ...}`

#### `ingest_pdf_directory(directory)` → Indexa una carpeta entera

Recorre todos los `.pdf` de la carpeta y llama a `ingest_pdf_file` para cada uno.

#### `get_statistics()` → Estadísticas para el dashboard

Consulta ChromaDB para saber cuántos documentos, fragmentos y páginas hay, y los agrupa por fuente.

---

### `src/ingestion/pdf_loader.py` — Lector de PDFs

Usa la librería `pypdf` para abrir archivos PDF y extraer su texto.

**Función `extract_pdf(path)`:**
- Abre el PDF con `PdfReader`
- Extrae el texto de cada página (si una página está corrupta, la salta sin abortar)
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

### `src/indexing/vector_store.py` — Base de datos vectorial

Es la interfaz entre el sistema y ChromaDB. Todos los demás módulos usan este archivo para guardar o buscar datos.

#### `get_collection()` — Obtiene la colección

Abre (o crea si no existe) la colección `"documentos_academicos"` en ChromaDB, configurada para usar distancia coseno.

#### `add_document(doc_id, chunks, metadata)` — Indexa fragmentos

Para cada fragmento del texto:
- Genera un ID único: `doc_id::chunk0`, `doc_id::chunk1`, etc.
- Le añade metadatos: título, autor, número de páginas, fecha de indexación, índice del fragmento
- Usa `upsert` (insert + update): si el documento ya estaba indexado, lo actualiza en lugar de duplicarlo

ChromaDB convierte automáticamente cada fragmento de texto en su vector (embedding) usando el modelo `all-MiniLM-L6-v2`.

#### `search(query, k=5)` — Búsqueda semántica

1. Convierte la consulta en un vector usando el mismo modelo de embeddings
2. Busca los `k` vectores más cercanos usando distancia coseno
3. Devuelve los fragmentos con su texto, metadatos y score de relevancia (0 a 1)

> **Score = 1 - distancia coseno.** Si el score es 0.95, el fragmento es muy relevante; si es 0.3, es poco relevante.

#### `_sanitize_metadata(meta)` — Limpia los metadatos

ChromaDB solo acepta valores de tipo `str`, `int`, `float` o `bool` en los metadatos. Esta función convierte listas a texto separado por comas y elimina los valores `None` para evitar errores.

#### `reset_collection()` — Borra todo

Elimina completamente la colección. Se usa cuando se quiere empezar desde cero.

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

**Función `generate(prompt, system)`:**
- Si `LLM_PROVIDER=groq` y hay `GROQ_API_KEY` → llama a la API de Groq (Llama 3 en la nube)
- Si `LLM_PROVIDER=ollama` → llama a un modelo local corriendo en tu máquina
- Si ninguno está disponible → devuelve `None` (el sistema usa respuesta extractiva)

**Modo extractivo (fallback):** Si no hay LLM, el sistema no falla. En lugar de generar texto con IA, simplemente muestra los 3 fragmentos más relevantes que encontró. Siempre responde algo útil.

---

### `src/rag/rag_pipeline.py` — Pipeline RAG completo

Combina la búsqueda semántica con la generación de respuesta.

**Función `answer(question, k=5)`:**

1. Llama a `vector_store.search(question, k=5)` → obtiene los 5 fragmentos más relevantes
2. Si no hay resultados → devuelve un mensaje indicando que no hay documentos
3. Construye el **contexto** concatenando los fragmentos con su fuente
4. Construye el **prompt** completo:
   ```
   CONTEXTO:
   [Fragmento 1 | Fuente: Nombre del PDF]
   texto del fragmento...

   [Fragmento 2 | Fuente: Otro PDF]
   texto del fragmento...

   PREGUNTA: ¿Cuál es la definición de X?

   RESPUESTA (basada solo en el contexto anterior):
   ```
5. Envía el prompt al LLM → recibe la respuesta generada
6. Si el LLM falla → usa la respuesta extractiva (muestra los fragmentos directamente)
7. Devuelve: `{answer, sources, provider}`

El **system prompt** le dice al LLM que solo responda con base en el contexto proporcionado y que si no sabe la respuesta, lo diga en lugar de inventar.

---

## 6. Carpeta `pages/` — las páginas del dashboard

Streamlit detecta automáticamente los archivos en esta carpeta y los convierte en páginas del menú lateral. El número al inicio del nombre (`1_`, `2_`, etc.) define el orden.

---

### `1_📥_Ingesta_de_Documentos.py` — Subir PDFs

Permite al usuario subir PDFs para indexarlos.

**Flujo del usuario:**
1. El usuario arrastra/selecciona uno o varios PDFs
2. Hace clic en "Procesar e indexar"
3. El sistema guarda el PDF en `data/raw/`, lo procesa con el pipeline y lo indexa en ChromaDB
4. Se muestra un resumen: cuántos fragmentos generó cada PDF
5. Se limpian las cachés de Streamlit para que el dashboard se actualice

También hay un botón para indexar todos los PDFs que ya estén en la carpeta `data/raw/` sin tener que subirlos otra vez.

---

### `2_💬_Chatbot_RAG.py` — Chatbot

Interfaz de chat para hacer preguntas en lenguaje natural sobre los documentos.

**Flujo:**
1. El usuario escribe una pregunta en el cuadro de chat
2. Se llama a `rag_pipeline.answer(pregunta, k=top_k)`
3. Se muestra la respuesta del LLM (o la extractiva si no hay LLM)
4. Se puede expandir "Fuentes" para ver qué fragmentos usó el sistema
5. El historial de la conversación se guarda en `st.session_state` (se pierde al recargar la página)

El slider en la barra lateral controla cuántos fragmentos recuperar (k = 1 a 10).

---

### `3_🔎_Búsqueda_Semántica.py` — Búsqueda directa

Permite buscar fragmentos por significado sin pasar por el LLM.

**Diferencia con el chatbot:** el chatbot genera una respuesta nueva usando los fragmentos + IA. La búsqueda semántica simplemente muestra los fragmentos más similares tal como están en los documentos, con su score de relevancia.

Muestra una tabla resumen y luego cada fragmento expandible con su texto completo y metadatos.

---

### `4_📊_Dashboard.py` — Estadísticas y clustering

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

---

## 7. Carpeta `scripts/` — herramientas CLI

### `run_pipeline.py` — Ejecutar el pipeline desde la terminal

Permite indexar PDFs directamente desde la consola sin abrir el dashboard. Útil para procesar muchos PDFs de una vez.

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
└── chroma/       ← base de datos vectorial (ver sección 2)
```

> La carpeta `data/` completa está en `.gitignore` — no se sube a GitHub. Cada persona que clone el proyecto empieza con la base de datos vacía y sube sus propios PDFs.

---

## 9. Flujo completo de datos

Este es el recorrido que hace un PDF desde que el usuario lo sube hasta que puede hacer preguntas sobre él:

```
Usuario sube un PDF
        │
        ▼
pages/1_📥_Ingesta_de_Documentos.py
  → guarda el archivo en data/raw/
  → llama a pipeline.ingest_pdf_file()
        │
        ▼
src/ingestion/pdf_loader.py
  → extract_pdf(): abre el PDF, extrae texto y metadatos
        │
        ▼
src/preprocessing/text_processing.py
  → clean_text(): limpia caracteres raros, normaliza espacios
  → chunk_text(): divide en fragmentos de ~900 chars con solapamiento
        │
        ▼
src/indexing/vector_store.py
  → add_document(): envía los fragmentos a ChromaDB
        │
        ▼
ChromaDB (internamente)
  → all-MiniLM-L6-v2 convierte cada fragmento en un vector de 384 números
  → guarda el vector en data_level0.bin
  → guarda el texto y metadatos en chroma.sqlite3
  → actualiza el grafo HNSW en link_lists.bin
        │
        ▼
Usuario hace una pregunta en el chatbot
        │
        ▼
src/rag/rag_pipeline.py → answer(question)
        │
        ├─→ src/indexing/vector_store.py → search()
        │     → ChromaDB convierte la pregunta en vector
        │     → busca los 5 vectores más cercanos (coseno)
        │     → devuelve fragmentos + scores
        │
        ├─→ construye el prompt con el contexto
        │
        └─→ src/rag/llm.py → generate(prompt)
              → Groq API (Llama 3) genera la respuesta
              → si no hay LLM → respuesta extractiva
                      │
                      ▼
              Se muestra la respuesta al usuario
              con las fuentes citadas
```

---

*Documentación generada para el Proyecto Integrador — Segundo Parcial · UTP · Grupo 7*
