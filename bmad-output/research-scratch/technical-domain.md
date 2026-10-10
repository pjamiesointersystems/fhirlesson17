# Technical and Domain Research: AI/LLM Integration with FHIR (2025-2026)

**Research Date:** October 7, 2026
**Purpose:** Inform restructuring of university-level lesson on "AI and FHIR" within a broader FHIR training course

---

## 1. Current State of AI + FHIR Integration (2025-2026)

### 1.1 Major Real-World Deployments

#### Epic Systems
Epic has deployed three primary AI product lines embedded throughout their EHR:
- **Emmie for Patients** -- assists patients with health information and engagement
- **Art for Clinicians** -- supports clinical documentation; reportedly enables "nurses [to] write notes 85% faster"
- **Penny for Revenue Cycle & Operations** -- automates financial and operational workflows

Epic integrates generative AI (including GPT-4) through a "HIPAA-compliant pipeline" within its EHR. The company partners with Microsoft on AI initiatives. Epic provides a public roadmap for "Generative AI and Cognitive Computing" features accessible through their UserWeb portal.

**Source:** https://www.epic.com/software/ai (accessed 2026-10-07)

#### Oracle Health (formerly Cerner)
Oracle Health has been integrating AI capabilities into its clinical platform. While direct documentation was unavailable via web fetch (404 errors on oracle.com/health URLs), Oracle's healthcare AI initiatives are known to leverage their broader cloud AI infrastructure. The LangCare MCP FHIR server explicitly lists Cerner as a supported EMR integration target using SMART on FHIR Backend Services authentication.

**Source:** Inferred from https://github.com/langcare/langcare-mcp-fhir (accessed 2026-10-07)

#### Mayo Clinic
Mayo Clinic Platform offers AI capabilities for healthcare data sharing and clinical research. Direct access to their platform site was restricted (403), but Mayo Clinic is known to be a significant player in clinical AI deployment, particularly through their data platform that enables AI model development on de-identified clinical data.

#### Google Cloud Healthcare API
Google Cloud provides a Healthcare API with FHIR store capabilities that explicitly supports "machine learning, AI, and other forms of data analytics" through its HTTPS protocol implementation. The platform also supports OMOP integration using Google's open-source Data Harmonization mapping engine for converting FHIR data to research standards -- a key pipeline for AI/ML research.

**Source:** https://docs.cloud.google.com/healthcare-api/docs/concepts/fhir (accessed 2026-10-07)

### 1.2 Key Frameworks and Standards

#### SMART on FHIR
SMART on FHIR remains the foundational API standard enabling substitutable healthcare applications. Key implementations include Apple Health Records API, HL7 FHIR API, Argonaut FHIR API, and CMS Blue Button 2.0. The 21st Century Cures Act made "APIs in EHRs the law," establishing regulatory requirements for interoperability.

Notable SMART ecosystem projects relevant to AI:
- **Cumulus** -- a "universal sidecar for learning healthcare systems" that uses AI/NLP to process FHIR data
- **CumulusQ** -- FHIR-native data quality tools
- **SMART Markers** -- patient-generated data framework

**Source:** https://smarthealthit.org (accessed 2026-10-07)

#### SMART Cumulus (AI-enabled FHIR Processing)
Cumulus is a containerized sidecar application funded by ONC that monitors healthcare data pipelines for a Learning Healthcare System. Key capabilities:
- Listens to the SMART/HL7 Bulk FHIR Access API
- Accesses US Core Data for Interoperability (over 100 FHIR data elements including clinical notes)
- Employs NLP and large language models to extract insights from unstructured clinical text
- Applications include symptom detection from physician notes, computable phenotype development, and text-to-FHIR conversion
- Operates securely behind provider firewalls with de-identification features
- Five health system-public health partnerships use Cumulus for disease surveillance (COVID-19, chronic diseases, opioid tracking, mental health)

**Source:** https://smarthealthit.org/cumulus (accessed 2026-10-07)

#### CDS Hooks
CDS Hooks implements a "hook-based pattern for invoking decision support from within a clinician's workflow." The system operates through real-time, workflow-triggered events such as:
- `patient-view` -- triggered when a patient record is opened
- `order-select` -- triggered during order entry
- `order-sign` -- triggered when signing orders

Services return CDS Cards (information, suggestion, or app link cards). The specification includes an OpenAPI/Swagger interface. CDS Hooks provides the natural integration point for AI-powered clinical decision support, as AI services can be registered as CDS Hook services that respond with evidence-based recommendations.

**Source:** https://cds-hooks.org (accessed 2026-10-07)

### 1.3 HL7/ONC Guidance on AI in Healthcare Interoperability

#### ONC HTI-1 Final Rule (January 2024)
The Health Data, Technology, and Interoperability: Certification Program Updates, Algorithm Transparency, and Information Sharing (HTI-1) Final Rule introduced significant AI-related provisions:
- **Decision Support Interventions (DSI):** New certification criteria for health IT that includes AI/ML-based clinical decision support
- **Predictive DSI:** Specific requirements for AI/ML models used in predictive analytics within certified health IT, including transparency requirements about the data used to train models, intended use, and known limitations
- **Algorithm Transparency:** Requirements for developers to provide source attributes and transparency information for DSI tools, enabling clinicians to understand the basis for AI-driven recommendations
- **USCDI Updates:** Expansion of the United States Core Data for Interoperability standard

Note: Direct access to the HTI-1 rule text was blocked by multiple federal website access restrictions (403/404 errors on healthit.gov and federalregister.gov).

**Source:** Referenced from https://www.healthit.gov (rule published January 9, 2024, effective date phased through 2025-2026)

#### FDA AI/ML Guidance
The FDA has authorized approximately 700+ AI-enabled medical devices for U.S. marketing as of September 2026. Key guidance documents include:
- "Good Machine Learning Practice for Medical Device Development: Guiding Principles"
- "Predetermined Change Control Plans for Machine Learning-Enabled Medical Devices: Guiding Principles"
- "Transparency for Machine Learning-Enabled Devices: Guiding Principles"
- FDA Digital Health and Artificial Intelligence Glossary

Device categories span radiology (dominant), cardiovascular, gastroenterology-urology, neurology, pathology, anesthesiology, and orthopedics. Notably, the FDA device list does not mention FHIR or interoperability standards, focusing instead on device-specific approvals.

**Source:** https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices (accessed 2026-10-07)

### 1.4 Notable Open-Source Projects Combining AI with FHIR

A GitHub search for "fhir llm" returned 224 repositories. The most notable projects (by stars and significance):

| Project | Stars | Language | Description |
|---------|-------|----------|-------------|
| **healthchainai/HealthChain** | 229 | Python | Healthcare AI SDK with typed, validated FHIR tools for agents |
| **StanfordBDHG/LLMonFHIR** | 158 | Swift | iOS app using LLMs to explain health records |
| **LifeValue/HealthWallet.me** | 71 | Dart | Patient-controlled health record app with on-device AI via FHIR R4 |
| **dermatologist/pyomop** | 67 | Python | OHDSI clinical data management with LLM queries, MCP server, FHIR import |
| **flexpa/llm-fhir-eval** | 46 | TypeScript | Benchmarking LLMs for FHIR tasks |
| **openmed-labs/openmed-agent** | 34 | -- | Terminal-native agent for clinical workflows with MCP and FHIR |
| **j-frei/Infherno** | 18 | Python | End-to-end agent for clinical notes to FHIR transformation |
| **hutchpd/AI-Medical-Scribe** | 16 | JavaScript | Browser-based AI medical scribe with local-first privacy |
| **Carer-Healthcare-AI/Hospilot** | 15 | Python | Open-source agentic AI operating layer for hospital operations |

**Source:** https://github.com/search?q=fhir+llm&type=repositories (accessed 2026-10-07)

#### HealthChain (Detail)
HealthChain is a Python SDK that addresses the gap where "Agents read clinical data fine -- writing it back correctly is the hard part." Key features:
- Type-safe FHIR resources with built-in validation including clinical code verification
- Multi-EHR data aggregation connecting to live FHIR APIs across Epic, Cerner, and other systems
- FHIRToolkit serving validated FHIR tools to agents via MCP or LangChain
- FHIR-grounded Q&A for patient-facing chatbots
- CLI scaffolding, Dockerized deployment, FastAPI integration
- Featured in TLDR AI Newsletter (900K+ developers), presented at NHS Python Open Source Conference
- Apache 2.0 licensed

**Source:** https://github.com/healthchainai/HealthChain (accessed 2026-10-07)

#### Stanford LLMonFHIR (Detail)
An iOS research application demonstrating LLM interpretation of FHIR health records:
- Accesses FHIR records from Apple Health
- Connects to OpenAI GPT API (GPT-4o, GPT-4o-mini)
- "Fog node" mode runs LLM inference on local network machines for privacy
- Built on Stanford's Spezi framework
- Uses FHIR as shared data repository between modules
- Supports synthetic Synthea-based patients for testing

**Source:** https://github.com/StanfordBDHG/LLMonFHIR (accessed 2026-10-07)

#### Flexpa LLM-FHIR-Eval (Detail)
Benchmarking framework for LLMs on FHIR tasks:
- Tests FHIR Resource Generation, Validation, Data Extraction, and Tool Use
- Models tested: GPT-3.5-turbo, GPT-4.1, O3 (low/high), Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude Sonnet 4, Claude Opus 4
- Evaluates extraction of structured FHIR data from unstructured clinical text
- Tests zero-shot and multi-turn tool interaction scenarios
- Supports recursive tool calling up to 10 depth levels for iterative refinement

**Source:** https://github.com/flexpa/llm-fhir-eval (accessed 2026-10-07)

#### Infherno (Detail)
End-to-end agent for transforming unstructured clinical notes into FHIR:
- Published as EACL 2026 System Demo paper
- Built on Hugging Face SmolAgents library
- Primary model: Gemini 2.5-Pro (recommended over local models for reliability)
- Uses RAG with SNOMED CT and HL7 ValueSets for semantic accuracy
- Public demo at infherno.misit-augsburg.de

**Source:** https://github.com/j-frei/Infherno (accessed 2026-10-07)

#### AI Medical Scribe (Detail)
Browser-based, local-first medical consultation capture:
- Live transcription, on-device summarization, document drafting
- Uses Chrome's built-in Prompt API (Gemini Nano) for on-device AI
- Exports as FHIR R4 JSON document Bundles
- AES-GCM encryption via Web Crypto API
- Requires Chrome with built-in AI flags, 22GB disk space, GPU with 4GB+ VRAM

**Source:** https://github.com/hutchpd/AI-Medical-Scribe (accessed 2026-10-07)

#### Hospilot (Detail)
Open-source agentic AI operating layer for hospital operations:
- **Fabric**: FHIR R5 translation layer ingesting hospital data via change APIs, polling, or Kafka
- **Agentic Framework**: Built on LangGraph with LLM planner converting natural-language goals into multi-agent pipelines
- 14 specialized domain agents (bed management, ER triage, ICU capacity, staffing, discharge planning, pharmacy, lab routing, OR scheduling, etc.)
- Human-in-the-loop approval gates for consequential actions
- Stack: PostgreSQL, Redis, LangGraph, Temporal, Anthropic Claude or OpenAI-compatible endpoints
- MIT licensed

**Source:** https://github.com/Carer-Healthcare-AI/Hospilot (accessed 2026-10-07)

---

## 2. Tools and Platforms for Local LLM Development with FHIR

### 2.1 LM Studio

**Current Status (2025-2026):**
- Python SDK version 1.5.0 (released August 22, 2025), with pre-release 1.6.0b1 available
- Requires Python 3.10+
- Provides OpenAI-compatible local API server
- GUI application for downloading and running GGUF-format models locally

**Key Features:**
- Synchronous `Client` component managing websocket connections
- Text completion and chat completion APIs
- Two usage approaches: convenience API (interactive) and direct Client (explicit connection control)
- Full type hints for API access verification
- Developer tooling: pytest, tox, Ruff linting, mypy strict mode

**SDK Installation:** `pip install lmstudio`

**Educational Suitability:** Good for beginners due to GUI interface. Students can visually browse and download models, then interact via the Python SDK. The OpenAI-compatible API means code written for LM Studio transfers to cloud APIs with minimal changes.

**Source:** https://pypi.org/project/lmstudio/ (accessed 2026-10-07)

### 2.2 Ollama

**Current Status (2025-2026):**
- Python SDK version 0.6.3 (released September 29, 2026)
- Requires Python 3.8+ (broader compatibility than LM Studio)
- MIT licensed

**Key Features:**
- Chat and text generation APIs
- Model management: pull, push, copy, delete, list models
- Text embeddings generation
- "System One" advanced decision-making with structured questions
- Synchronous and AsyncClient options
- Cloud integration via ollama.com with API authentication
- Uses httpx for HTTP operations

**SDK Installation:** `pip install ollama`
**Prerequisite:** Ollama application must be installed and running; models pulled via `ollama pull <model>`

**Educational Suitability:** Excellent for command-line-oriented students. Simpler setup than LM Studio for automated/scripted workflows. The CLI-first approach teaches model management fundamentals. AsyncClient support is valuable for building responsive applications.

**Source:** https://pypi.org/project/ollama/ (accessed 2026-10-07)

### 2.3 llama-cpp-python

**Current Status:**
- Version 0.3.36
- Python bindings for the llama.cpp C++ library
- Requires Python 3.8+

**Key Features:**
- High-level API: text completion, chat completion, function calling
- JSON and JSON Schema constrained outputs (important for FHIR generation)
- Multi-modal support (LLaVA, Moondream2, MiniCPM-V, Qwen2.5-VL)
- Speculative decoding for faster inference
- OpenAI-compatible web server included
- Hardware acceleration: CUDA (11.8-13.2), Metal (macOS), ROCm/HIP (AMD), Vulkan, SYCL
- Direct Hugging Face Hub model downloading (GGUF format)
- LangChain and LlamaIndex integration

**Educational Suitability:** Most technically demanding option. Best for advanced students who want to understand the inference stack. JSON Schema constrained output is particularly valuable for generating valid FHIR resources. The OpenAI-compatible server can serve as a drop-in local replacement for cloud APIs.

**Source:** https://pypi.org/project/llama-cpp-python/ (accessed 2026-10-07)

### 2.4 GPT4All

GPT4All is a desktop application for running local LLMs with a focus on privacy. It provides:
- Cross-platform GUI application
- Python SDK for programmatic access
- Support for various open-source models
- On-device processing with no data leaving the machine
- Document Q&A with local RAG capabilities

Note: Direct documentation access was blocked (certificate errors on gpt4all.io and docs.gpt4all.io).

**Source:** https://gpt4all.io (access attempted 2026-10-07, blocked by certificate error)

### 2.5 Jan

Jan is an open-source desktop application for running local LLMs. It provides:
- Cross-platform GUI (Windows, Mac, Linux)
- OpenAI-compatible API server
- Model marketplace for downloading models
- Extensions system for customization

Note: Direct documentation access was blocked (certificate error on jan.ai).

**Source:** https://jan.ai (access attempted 2026-10-07, blocked by certificate error)

### 2.6 Comparison for Educational/Training Purposes

| Feature | LM Studio | Ollama | llama-cpp-python | GPT4All |
|---------|-----------|--------|-----------------|---------|
| **Setup Difficulty** | Easy (GUI) | Easy (CLI) | Moderate (build) | Easy (GUI) |
| **Python SDK** | lmstudio 1.5.0 | ollama 0.6.3 | llama-cpp-python 0.3.36 | gpt4all |
| **Min Python** | 3.10 | 3.8 | 3.8 | 3.8 |
| **OpenAI-compatible API** | Yes | Yes (via API) | Yes (built-in server) | Partial |
| **JSON Schema Constrained Output** | Via API | Via API | Native support | Limited |
| **GPU Support** | Auto-detect | Auto-detect | Manual config | Auto-detect |
| **Model Format** | GGUF | GGUF (via Modelfile) | GGUF | GGUF |
| **Best For** | Visual learners, beginners | CLI users, automation | Advanced users, custom pipelines | Privacy-focused users |
| **FHIR Generation** | Good (via prompting) | Good (via prompting) | Best (constrained output) | Basic |

**Recommendation for Training:** For a university-level FHIR course, **Ollama** offers the best balance of simplicity and capability. Its CLI-first approach teaches fundamentals, the Python SDK is straightforward, and it supports the broadest Python version range. **LM Studio** is a good secondary option for students who prefer a GUI. **llama-cpp-python** should be presented as the advanced option, highlighting its JSON Schema constrained output capability for generating structurally valid FHIR resources.

---

## 3. Emerging Trends

### 3.1 MCP (Model Context Protocol) Servers for EHR/FHIR Integration

The Model Context Protocol (MCP) is an open-source standard from Anthropic for connecting AI applications to external systems. It provides a standardized way for AI agents to access data sources, tools, and workflows. MCP is supported by Claude, ChatGPT, VS Code, Cursor, and many other AI platforms.

**Source:** https://modelcontextprotocol.io (accessed 2026-10-07)

#### FHIR MCP Servers (Ranked by Stars)

| Project | Stars | Language | Key Features |
|---------|-------|----------|-------------|
| **wso2/fhir-mcp-server** | 137 | Python | SMART-on-FHIR auth, FHIRPath filtering, multi-transport (HTTP, SSE, STDIO) |
| **the-momentum/fhir-mcp-server** | 102 | Python | Full CRUD, RAG services, Pinecone vector store, LOINC integration |
| **langcare/langcare-mcp-fhir** | 67 | Go | Enterprise-grade, Epic/Cerner/OpenEMR support, PHI scrubbing, MCP Apps UI |
| **aks129/HealthClawGuardrails** | 30 | Python | Security guardrails, PHI redaction, audit trails, human-in-the-loop |
| **erikhoward/azure-fhir-mcp-server** | 18 | Python | Azure Health Data Services integration |
| **faulkj/fhirHydrant** | 18 | TypeScript | SMART Backend Services, FHIRPath filtering |

**Source:** https://github.com/search?q=mcp+server+fhir&type=repositories (accessed 2026-10-07)

#### WSO2 FHIR MCP Server (Detail)
- Exposes any FHIR server as an MCP server
- SMART-on-FHIR authentication with OAuth 2.0 Authorization Code Grant
- Response filtering using FHIRPath expressions
- Multi-transport: Streamable HTTP (default port 8000), SSE, STDIO
- Deployable via PyPI, source, Docker, or Docker Compose
- Includes pre-configured HAPI FHIR server for testing
- Integrates with VS Code, Claude Desktop, and MCP Inspector

**Source:** https://github.com/wso2/fhir-mcp-server (accessed 2026-10-07)

#### Momentum FHIR MCP Server (Detail)
- Full CRUD operations on FHIR resources
- RAG services with document processing and semantic embeddings
- Pinecone-based vector store for similarity search
- LOINC terminology lookups and validation
- AI-powered document ingestion (TXT, CSV, JSON, PDF)
- Encryption for sensitive configuration values
- Docker deployment support

**Source:** https://github.com/the-momentum/fhir-mcp-server (accessed 2026-10-07)

#### LangCare MCP FHIR (Detail)
- Enterprise-grade Go implementation
- Four generic FHIR operations: fhir_read, fhir_search, fhir_create, fhir_update
- Integrates with Epic (private_key_jwt), Cerner (SMART Backend), OpenEMR (OAuth2), GCP Healthcare API
- Two-layer auth: client auth + backend auth with automatic token refresh
- TLS 1.3, PHI scrubbing in logs, mTLS, HIPAA-aligned audit logging
- Stateless proxy -- no persistent PHI storage
- MCP Apps: embedded React UIs (FHIR Explorer, Patient Chart Review)
- 40+ clinical skills/workflow guides
- Deployable locally (stdio), HTTP, or remote (Fly.io)

**Source:** https://github.com/langcare/langcare-mcp-fhir (accessed 2026-10-07)

#### HealthClaw Guardrails (Detail)
- Security proxy between AI agents and FHIR servers
- PHI redaction: names to initials, identifiers stripped, addresses removed, DOB to year-only
- Immutable AuditEvent audit trail
- Step-up HMAC-SHA256 authorization for writes
- Human-in-the-loop approval for real-world actions
- Tenant isolation with cross-tenant blocking
- Self-test conformance harness (Grade A-F scoring)
- 29 MCP tools (read/write, lab interpretation, care-gaps, form population)
- Supports Claude plugins, OpenAI/Gemini adapters, HTTP bridges
- FHIR R4 US Core v9 and R6 v6.0.0-ballot3
- MIT licensed, 934 commits, 1490+ Python tests

**Source:** https://github.com/aks129/HealthClawGuardrails (accessed 2026-10-07)

### 3.2 Agentic AI Workflows in Healthcare

Agentic AI -- where AI systems autonomously plan and execute multi-step tasks -- is rapidly emerging in healthcare FHIR applications:

**Hospilot** exemplifies the agentic approach with 14 specialized domain agents coordinated by an LLM planner via LangGraph. The system converts natural-language goals into directed acyclic graphs of agent actions, with human approval gates for consequential decisions.

**OpenMed Agent** provides a terminal-native agentic interface for clinical workflows including prior authorization, ICD-10 coding audits, clinical documentation transformation, and care coordination. It supports GPT-6.1 Sol, Claude, and custom OpenAI-compatible servers, with optional MCP connections for tool extensibility.

**HealthChain** serves as the SDK layer enabling agentic development, providing typed FHIR tools to agents via MCP or LangChain with built-in validation to ensure agent-written clinical data is valid and correctly coded.

**Key Pattern:** The emerging architecture is: **LLM Agent -> MCP Server -> FHIR Server -> EHR**, with guardrail layers (like HealthClaw) inserted between the MCP server and FHIR server for security.

### 3.3 FHIR-Native AI Agents

A new class of AI agents is emerging that treats FHIR as a first-class data model rather than an afterthought:

- **Infherno** uses RAG grounded in SNOMED CT and HL7 ValueSets to convert clinical notes to validated FHIR resources
- **HealthChain** provides type-safe FHIR resource generation with clinical code validation
- **LLM-FHIR-Eval** benchmarks demonstrate that modern LLMs (Claude Opus 4, GPT-4.1, O3) can generate valid FHIR resources and extract structured FHIR data from unstructured text
- **AI Medical Scribe** generates FHIR R4 document Bundles directly from transcribed consultations

### 3.4 RAG (Retrieval-Augmented Generation) with FHIR Data

RAG architectures combining FHIR data with LLMs are a significant trend:

- **Momentum FHIR MCP Server** includes dedicated RAG services with document processing, semantic embeddings, and Pinecone vector store integration for similarity search over clinical data
- **Infherno** uses RAG to ground extracted medical concepts in SNOMED CT and HL7 ValueSets
- **SMART Cumulus** employs NLP and LLMs with Bulk FHIR data for population-level insights
- **Stanford LLMonFHIR** retrieves FHIR records and provides them as context for LLM-generated explanations

**Common RAG Pattern for FHIR:**
1. Extract FHIR resources via FHIR API (or Bulk FHIR for population data)
2. Chunk and embed resources into vector store
3. At query time, retrieve relevant FHIR resources via semantic search
4. Provide retrieved resources as context to LLM for generating responses
5. Validate any generated FHIR output against schema and terminology

### 3.5 AI-Powered Clinical Decision Support Using FHIR

The integration of AI with CDS Hooks provides a natural pathway for AI-powered clinical decision support:

1. **CDS Hooks + AI Services:** AI models can be registered as CDS Hook services, responding to clinical workflow events (patient-view, order-select, order-sign) with AI-generated recommendations returned as CDS Cards
2. **SMART Cumulus:** AI-powered surveillance and phenotyping using Bulk FHIR data behind provider firewalls
3. **HealthClaw Guardrails:** Provides 29 MCP tools including lab interpretation and care-gaps reminders, with security guardrails for AI-driven CDS
4. **Hospilot:** 14 specialized agents for operational decision support (bed management, triage, staffing, discharge planning)

---

## 4. Privacy and Security

### 4.1 HIPAA Considerations for AI with FHIR Data

HIPAA remains the primary regulatory framework governing AI use with health data. Key considerations:

- **Business Associate Agreements (BAAs):** Any AI vendor processing PHI on behalf of a covered entity must sign a BAA. This applies to cloud LLM providers (OpenAI, Anthropic, Google) when PHI is sent to their APIs.
- **Minimum Necessary Standard:** AI systems should access only the minimum FHIR resources necessary for their function
- **Access Controls:** AI systems accessing FHIR servers must implement proper authentication (SMART on FHIR, OAuth 2.0) and authorization scoping
- **Audit Requirements:** All AI access to PHI must be logged; projects like HealthClaw Guardrails generate immutable AuditEvent records for every access

Note: Direct access to HHS HIPAA AI guidance pages was blocked (403 errors on hhs.gov).

**Source:** General HIPAA requirements applied to AI context; https://github.com/aks129/HealthClawGuardrails (accessed 2026-10-07)

### 4.2 Local vs. Cloud LLM Deployment for PHI

**Critical distinction for FHIR training:** The choice between local and cloud LLM deployment has significant HIPAA implications:

#### Cloud LLM Deployment
- Requires BAA with the LLM provider
- Epic's approach: HIPAA-compliant pipeline with Microsoft/OpenAI under BAA
- Advantages: More powerful models, regular updates, managed infrastructure
- Risks: Data leaves institutional control, potential for model training on PHI (must be contractually prohibited)

#### Local LLM Deployment
- PHI never leaves the institution's infrastructure
- Tools: Ollama, LM Studio, llama-cpp-python, GPT4All
- Stanford LLMonFHIR's "fog node" mode demonstrates local inference for privacy
- AI Medical Scribe uses Chrome's Gemini Nano for entirely on-device processing
- SMART Cumulus operates behind provider firewalls
- Advantages: Full data control, no BAA needed for the LLM component, air-gapped option
- Limitations: Smaller models, less capable than cloud alternatives, requires local GPU resources
- **Educational opportunity:** This trade-off is ideal for classroom discussion

#### Hybrid Approaches
- De-identify FHIR data before sending to cloud LLMs
- Use local models for PHI-containing tasks, cloud models for non-PHI tasks
- LangCare MCP FHIR implements stateless proxy design with "no persistent PHI storage"
- HealthClaw Guardrails implements PHI redaction (names to initials, identifiers stripped) before agent access

### 4.3 De-identification Approaches

#### Microsoft FHIR Tools for Anonymization
Open-source toolkit for de-identifying FHIR and DICOM data:
- Configuration-based approach with customizable anonymization parameters
- Sample configuration based on HIPAA Safe Harbor de-identification standard
- Built on .NET frameworks
- Does not access, collect, or manage user data -- fully local processing
- MIT licensed

**Source:** https://github.com/microsoft/FHIR-Tools-for-Anonymization (accessed 2026-10-07)

#### HealthClaw Guardrails PHI Redaction
Implements real-time PHI redaction at the proxy layer:
- Names reduced to initials
- Identifier values stripped
- Addresses removed
- Birth dates truncated to year-only
- Applied automatically before AI agent access

**Source:** https://github.com/aks129/HealthClawGuardrails (accessed 2026-10-07)

#### HIPAA Safe Harbor Method
The standard de-identification approach removes 18 categories of identifiers:
- Names, geographic data smaller than state, dates (except year), phone/fax numbers, email addresses, SSN, MRN, health plan numbers, account numbers, certificate/license numbers, VIN/serial numbers, device identifiers, URLs, IP addresses, biometric identifiers, full-face photos, and any other unique identifying number

#### Synthetic Data for Training
For educational purposes, several projects provide synthetic FHIR data:
- **Synthea** -- generates realistic synthetic FHIR patient records
- **Stanford LLMonFHIR** supports Synthea-based synthetic patients for testing
- **HealthClaw Guardrails** provides a public demo with synthetic data at their Railway-hosted endpoint

### 4.4 ONC/FDA Guidance on AI in Clinical Settings

#### ONC HTI-1 Final Rule (2024)
Key AI-related provisions:
- Introduced "Decision Support Interventions" (DSI) as a certification criterion replacing the older CDS criterion
- **Predictive DSI** subcategory specifically addresses AI/ML models used in clinical prediction, requiring:
  - Transparency about training data sources
  - Description of intended use and known limitations
  - Information about model validation and performance
  - Risk management practices
- Developers must provide "source attributes" enabling end-users to understand AI recommendations
- Effective dates phased through 2025-2026

#### FDA Framework
- 700+ AI/ML-enabled medical devices authorized as of September 2026
- Focus areas: radiology (dominant), cardiovascular, neurology, pathology
- Key guidance: Good ML Practice, Predetermined Change Control Plans, Transparency Principles
- FDA treats AI/ML clinical tools as Software as a Medical Device (SaMD) when they provide diagnostic or therapeutic recommendations
- Notably, FDA's device authorization list does not reference FHIR or interoperability standards

#### Key Regulatory Gap
There is currently no explicit regulatory requirement linking FHIR interoperability standards to AI/ML model deployment. The HTI-1 rule addresses AI transparency within certified health IT (which must support FHIR APIs), but does not mandate FHIR-specific AI standards. This represents both a gap and an opportunity for the field.

**Sources:**
- https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices (accessed 2026-10-07)
- ONC HTI-1 Final Rule (published January 9, 2024 in the Federal Register)

---

## 5. Summary of Key Findings for Lesson Design

### The AI+FHIR Ecosystem Has Matured Significantly
- From conceptual in 2023 to production deployments in 2025-2026
- 224+ GitHub repositories combining FHIR and LLM capabilities
- Multiple production MCP servers bridging AI agents to FHIR servers
- Major EHR vendors (Epic, Oracle Health) actively deploying AI features

### MCP Is the Dominant Integration Pattern
- Model Context Protocol has become the standard way to connect AI agents to FHIR servers
- Six significant FHIR MCP server implementations exist, ranging from simple proxies to enterprise-grade solutions
- The architecture pattern is well-established: AI Agent -> MCP -> FHIR Server -> EHR

### Local LLMs Are Viable for Education
- Ollama and LM Studio both provide mature, well-documented Python SDKs
- llama-cpp-python offers advanced features like JSON Schema constrained output for FHIR generation
- Local deployment eliminates HIPAA concerns for training scenarios
- Multiple tools (AI Medical Scribe, LLMonFHIR fog mode) demonstrate local-first AI with FHIR

### Regulatory Framework Is Evolving
- HTI-1 introduced DSI/Predictive DSI transparency requirements
- FDA has authorized 700+ AI medical devices but without FHIR-specific standards
- HIPAA applies to all AI processing of PHI; local LLMs offer a compliance-friendly path
- Gap exists between AI regulation and FHIR interoperability standards

### Key Open-Source Projects for Teaching
1. **HealthChain** -- best SDK for building FHIR-aware AI agents (Python)
2. **WSO2 FHIR MCP Server** -- best for demonstrating MCP+FHIR integration
3. **LLM-FHIR-Eval** -- benchmarking framework showing LLM FHIR capabilities
4. **HealthClaw Guardrails** -- demonstrates security/privacy patterns
5. **Infherno** -- shows clinical note to FHIR transformation pipeline

---

## Appendix: All Sources Consulted

| URL | Status | Date Accessed |
|-----|--------|---------------|
| https://www.epic.com/software/ai | Success | 2026-10-07 |
| https://smarthealthit.org | Success | 2026-10-07 |
| https://smarthealthit.org/cumulus | Success | 2026-10-07 |
| https://cds-hooks.org | Success | 2026-10-07 |
| https://modelcontextprotocol.io | Success | 2026-10-07 |
| https://pypi.org/project/ollama/ | Success | 2026-10-07 |
| https://pypi.org/project/lmstudio/ | Success | 2026-10-07 |
| https://pypi.org/project/llama-cpp-python/ | Success | 2026-10-07 |
| https://github.com/search?q=fhir+llm&type=repositories | Success | 2026-10-07 |
| https://github.com/search?q=mcp+server+fhir&type=repositories | Success | 2026-10-07 |
| https://github.com/healthchainai/HealthChain | Success | 2026-10-07 |
| https://github.com/StanfordBDHG/LLMonFHIR | Success | 2026-10-07 |
| https://github.com/flexpa/llm-fhir-eval | Success | 2026-10-07 |
| https://github.com/j-frei/Infherno | Success | 2026-10-07 |
| https://github.com/openmed-labs/openmed-agent | Success | 2026-10-07 |
| https://github.com/Carer-Healthcare-AI/Hospilot | Success | 2026-10-07 |
| https://github.com/hutchpd/AI-Medical-Scribe | Success | 2026-10-07 |
| https://github.com/wso2/fhir-mcp-server | Success | 2026-10-07 |
| https://github.com/the-momentum/fhir-mcp-server | Success | 2026-10-07 |
| https://github.com/langcare/langcare-mcp-fhir | Success | 2026-10-07 |
| https://github.com/aks129/HealthClawGuardrails | Success | 2026-10-07 |
| https://github.com/microsoft/FHIR-Tools-for-Anonymization | Success | 2026-10-07 |
| https://github.com/microsoft/FHIR-Converter | Success | 2026-10-07 |
| https://docs.cloud.google.com/healthcare-api/docs/concepts/fhir | Success | 2026-10-07 |
| https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices | Success | 2026-10-07 |
| https://www.healthit.gov/topic/artificial-intelligence | No content | 2026-10-07 |
| https://www.healthit.gov/* (multiple paths) | 404/403 | 2026-10-07 |
| https://www.hhs.gov/hipaa/for-professionals/* | 403 | 2026-10-07 |
| https://confluence.hl7.org/display/FHIR/Artificial+Intelligence | 405 | 2026-10-07 |
| https://lmstudio.ai | Cert error | 2026-10-07 |
| https://ollama.com | Cert error | 2026-10-07 |
| https://gpt4all.io | Cert error | 2026-10-07 |
| https://jan.ai | Cert error | 2026-10-07 |
| https://medplum.com | Cert error | 2026-10-07 |
| https://www.federalregister.gov/* | Blocked | 2026-10-07 |
