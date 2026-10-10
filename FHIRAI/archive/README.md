# Archive — Lesson 17 Prior Development Files

This directory contains files from prior development iterations of Lesson 17. They are preserved for reference but are not actively maintained or part of the current lesson structure.

**Archived:** 2026-10-09 (Story 3.4 — File Cleanup and Reorganization)

## Undocumented Summarization Apps

- **FHIRSummaryApp_by_resource_type.py** — Patient summarization using resource-type grouping. Concepts absorbed into Exercise 3 (`exercise3_summarizer.py`).
- **FHIRSummaryRevised.py** — Revised summarization app. Concepts absorbed into Exercise 3.
- **OpenAIFHIRSummary.py** — Cloud LLM summarization variant using OpenAI API. Out of scope for Lesson 17 (local models only).
- **fhirsummaryapp.py** — Earlier summarization app variant.
- **simplesummary.py** — Simple summarization script.

## Superseded Files

- **fhirapp.py** — Original Textual TUI application. Replaced by the Exercise 3 CLI pipeline. Textual was removed from the lesson dependencies.
- **getSearchPatients.py** — Original FHIR client module. Refactored into `fhir_client.py` (Story 1.3) with updated imports, externalized config, and Pydantic v2 compatibility.
- **llamachat.py** — LM Studio connectivity test. Replaced by Exercise 1 (`exercise1_llm_chat.py`) which supports both Ollama and LM Studio.
- **main.py** — Minimal entry point script, no longer needed.

## RAG-Related Files (Deferred to RAG Lab)

These files implement RAG pipelines, vector stores, or embeddings — topics covered in the RAG Lab lesson, not Lesson 17. They are preserved here until the RAG Lab repository is established.

- **fhirvectorflattened.py** — RAG pipeline with vector flattening
- **simplevector.py** — Simple vector store implementation
- **simplevectorstorage.py** — Vector storage utilities
- **testrag.py** — RAG test script
- **fhir_rag_chat.py** — FHIR RAG chat interface

## Deleted Files (Not Archived)

The following files were deleted outright (not preserved):

- **testdriver.py** — Had bugs, duplicated testrag.py functionality
- **testpydriver.py** — IRIS connection test, not educational content
- **rag_summary.log** — Log output from RAG experiments
- **simplevectorstorage.bak** — Backup of vector storage file

## Other Archived Files

- **LM Studio.docx** — LM Studio setup documentation (superseded by Readme.md setup instructions)
- **pyproject.toml** — Previous project configuration file
- **uv.lock** — Previous uv package manager lock file
- **test.http** — HTTP test requests file

## Usage Note

These files are not runnable without updates to match current dependencies (fhir.resources 7.x+, Pydantic v2, Textual removed, Ollama replacing LM Studio as primary). They are provided as historical reference only.
