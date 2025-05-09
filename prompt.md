Prompt definitivo (ES) — versión aclarada
Objetivo
Diseña un pipeline en Python 3.12 basado en la última versión de LangChain que procese lotes de PDF agrupados por carpetas y genere, para cada carpeta, un informe unificado de alta fidelidad. El código debe ser robusto, legible y de calidad profesional, cumpliendo buenas prácticas de tipado, modularidad, pruebas unitarias y registro exhaustivo. No optimices costes de inferencia; prioriza la precisión y la utilidad del resultado.

Alcance funcional
Entrada

Directorio raíz: documents/.

Cada subcarpeta inmediata (carpeta1/, carpeta2/, …) representa un contexto independiente.

Todos los PDF contenidos en una subcarpeta se fusionan en un único context para el LLM que elaborará el informe de esa carpeta.

Flujo por PDF

2.1 Extracción y NLP
Extraer texto con PyPDF2 (o biblioteca equivalente, libre de dependencias nativas).

Ejecutar spaCy (en_core_web_sm) para detectar entidades:

ORG (organizaciones)

GPE (entidades geopolíticas)

2.2 Normalización y clasificación de entidades
Actores / Stakeholders (ORG)

Tomar la lista de ORG detectadas por spaCy.

Enviar esa lista al LLM con un prompt que debe:

Normalizar ortografía, mayúsculas y variantes.

Eliminar duplicados y falsos positivos (URLs, genéricos, etc.).

Con la lista limpia, pedir al mismo prompt que compare cada organización contra el taxonomy de actores (JSON suministrado).

Si existe coincidencia ( fuzzy match ≥ 90 % ), retornar el nombre normalizado y su categoría de la taxonomía.

Si no existe coincidencia, instruir al LLM para que busque explícitamente en el texto otros actores/stakeholders relevantes y los incluya, indicando “No clasificado” como categoría.

El resultado final es una lista de objetos:

json
Copiar
Editar
[
  { "nombre": "European Commission", "categoria": "Political Actors" },
  { "nombre": "World Bank", "categoria": "Economic Actors" },
  { "nombre": "Iniciativa X", "categoria": "No clasificado" }
]
Insertar esta lista en la sección “Actores / Stakeholders” del informe, formateada como tabla Markdown o lista enriquecida.

Ubicaciones (GPE)

Enviar la lista de GPE al LLM con otro prompt que debe:

Normalizar (p. ej. “USA” → “United States”).

Eliminar duplicados y entidades no geográficas.

Tras la normalización, devolver un string con las ubicaciones separadas por comas:

graphql
Copiar
Editar
Argentina, Brazil, European Union, Germany
Colocar esta línea en una sección separada llamada “Ubicaciones”, situada inmediatamente después de la sección “Fecha” y antes de “Resumen Ejecutivo”.

2.3 Extracción semántica de campos por documento
Para cada PDF, generar mediante prompts específicos:

Título (conciso y descriptivo)

Fecha (formato preciso YYYY-MM-DD; degradar a YYYY-MM)

Resumen Ejecutivo (≤ 150 palabras)

Características clave (3-6 viñetas ≤ 30 palabras c/u)

Temas principales

Enviar al LLM el texto fuente y el taxonomy de temas (JSON).

El LLM debe:

Identificar los temas mencionados en el texto.

Para cada tema identificado, indicar la categoría exacta dentro del taxonomy; si un tema no figura, clasificarlo como “Tema no clasificado”.

El resultado es un objeto JSON:

json
Copiar
Editar
{
  "Technology & Innovation": ["Artificial Intelligence", "Blockchain"],
  "Data & Governance": ["Cybersecurity"],
  "Tema no clasificado": ["Green IT"]
}
Mostrarlo en la sección “Temas principales” como lista jerárquica.

Aplicaciones prácticas ya implementadas (solo iniciativas existentes)

Compromisos futuros cuantificables (promesas con metas, montos o fechas claras)

2.4 Evaluación de fidelidad
Calcular un faithfulness score (0-100) con langchain.evaluation y un juez GPT-4-o, comparando el texto fuente contra el resumen de temas.

Incluir el puntaje y la calificación (“excelente”, “regular”, “malo”) al inicio del informe.

Salida por carpeta

outputs/<carpeta>.md – informe consolidado en Markdown.

outputs/<carpeta>.docx – conversión automática con Pandoc.

outputs/<carpeta>.json – versión estructurada con todos los campos y el faithfulness score.

CLI
Ejecutable process_pdfs.py con flags:

bash
Copiar
Editar
--provider {openai,gemini}
--model <modelo>
--in <ruta_entrada>    # default: documents/
--out <ruta_salida>    # default: outputs/
--log-level {DEBUG,INFO,WARNING,ERROR}
--max-workers <int>    # paralelizar carpetas
Requisitos no funcionales
Arquitectura hexagonal: domain, services, interfaces.

Tipado estricto (mypy --strict) y modelos Pydantic v2.

Manejo de errores granular y registro estructurado con structlog.

Validación de schema (jsonschema) antes de persistir JSON.

Paralelización opcional (ThreadPoolExecutor) y caché LangChain.

Tests (pytest, > 90 % cobertura) y pre-commit hooks (black, ruff, isort, mypy, pytest).

Dependencias gestionadas con Poetry; versiones fijadas.

Guía de despliegue (README.md) y documentación (mkdocs opcional).

Mejoras y extensiones sugeridas
Chunking adaptativo + recuperación semántica.

Tracing con LangSmith.

Persistencia en SQLite/DuckDB para analítica posterior.

Modo dry-run para pruebas rápidas.

CI/CD con GitHub Actions.

Entregables
Código fuente estructurado.

poetry.lock + requirements.txt.

Ejemplos en sample_outputs/.

Reporte de cobertura (htmlcov/).

Documentación (docs/).

Importante
En esta fase solo se entrega el código anterior como referencia; este prompt es la especificación exhaustiva que el equipo de desarrollo debe implementar.