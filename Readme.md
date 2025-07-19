# 🧠 Using LLMs with FHIR – Concepts, Use Cases & Foundations

This repository accompanies **Lesson 17** of the FHIR course and provides hands-on materials for exploring how **Large Language Models (LLMs)** can be used with **FHIR** (Fast Healthcare Interoperability Resources) to enable powerful healthcare applications.

Created by **Patrick W. Jamieson, M.D.**, Technical Product Manager, with contributions from **Russ Leftwich, M.D.**, Senior Clinical Advisor, Interoperability at InterSystems.

---

## 🎯 Learning Objectives

By engaging with this lesson and repository, learners will be able to:

- Understand the significance of LLMs in healthcare
- Identify key components and architectures of LLMs
- Recognize the trade-offs between local and hosted LLM deployments
- Run local LLMs (e.g. via [LM Studio](https://lmstudio.ai)) for FHIR-based summarization tasks
- Build foundational skills in prompt engineering
- Explore real-world use cases of combining LLMs with FHIR
- Identify hallucination risks in LLMs and how to mitigate them

---

## 🧰 Repository Contents

- `phenoml_client.py` – A Python client for interacting with a PhenoML-compatible API that supports:
  - Authentication
  - Creating and searching FHIR resources from natural language
  - Managing AI agents and prompts
  - Extracting clinical codes from text (e.g., ICD-10-CM)
- `README.md` – This file.
- `.env.example` – Sample template for API credentials and environment variables (rename to `.env`).
- Presentation: `Using LLMS with FHIR.pptx` – Slide deck introducing core concepts and healthcare use cases.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/pjamiesointersystems/fhirlesson17.git
cd fhirlesson17
This lesson demonstrates how to run a language model locally using LM Studio. Steps include:

Download and install LM Studio.

Browse and load a quantized LLM (e.g., Mistral or Phi) from Hugging Face.

Start the LM Studio API Server.

Connect and query the model using Python or the built-in chat UI.

This provides a privacy-preserving way to run LLM-powered summarization or analysis on local FHIR bundles, such as those retrieved via the FHIR Patient/$everything operation.

💡 Real-World Use Cases Explored
Summarizing patient records to assist diagnosis (e.g., Mayo Clinic)

Generating SOAP notes from clinical conversation transcripts (e.g., Abridge, Amazon HealthScribe)

Natural language querying of FHIR or SQL (e.g., “Find all diabetics with HbA1c > 8”)

Extracting structured data from free-text notes into FHIR

Clinical trial matching and registry reporting

Literature summarization (e.g., PubMed GPT, BioGPT)

Patient self-triage with AI chatbots (e.g., Ada Health, MedPaLM)

⚠️ LLM Hallucinations in FHIR
LLMs may generate incorrect or non-compliant FHIR queries. To mitigate this:

Use grounded prompts that include FHIR spec examples

Apply retrieval-augmented generation (RAG)

Fine-tune on validated FHIR corpora

Involve human-in-the-loop for review

Choose specialized models when possible

📚 Additional Resources
FHIR Overview – HL7

Hugging Face Transformers

LM Studio App

InterSystems IRIS for Health

🏁 License
This project is intended for educational purposes. All trademarks referenced are the property of their respective owners.

🙏 Acknowledgments
Based on the lecture and slide materials from Using LLMs with FHIR by Patrick W. Jamieson and Russ Leftwich.




---

Let me know if you'd like this README file formatted for Canvas or if you'd like to include instructions for running in Google Colab.



