from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Markdown, ProgressBar
from textual.containers import VerticalScroll
from textual.worker import get_current_worker
from openai import OpenAI
import json, decimal, tiktoken, asyncio, sys, os, re
import iris
from sentence_transformers import SentenceTransformer

RESOURCE_TYPES = [
    "Patient", "Condition", "MedicationRequest", "Observation",
    "Encounter", "Practitioner", "Procedure", "AllergyIntolerance",
    "Immunization", "DiagnosticReport", "DocumentReference", "CarePlan"
]

class FHIRSummaryAppOpenAI(App):
    BINDINGS = [('d', 'toggle_dark', 'Toggle dark mode')]

    def __init__(self, ptFHIRid, **kwargs):
        super().__init__(**kwargs)
        self.fhirId = ptFHIRid
        self.embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        self.conn = iris.connect("127.0.0.1", 1972, "DEMO", "_SYSTEM", "ISCDEMO")
        self.partial_summaries = {}
        self.final_summary_text = ""
        self.client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def log_to_file(self, message: str, file_path: str = "rag_summary.log") -> None:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with VerticalScroll():
            yield Markdown("# Partial Summaries", id="partial-summary-header")
            self.progress = ProgressBar(total=len(RESOURCE_TYPES) + 1, id="progress-bar")
            yield self.progress
            self.summary_widgets = []

    async def on_mount(self) -> None:
        open("rag_summary.log", "w", encoding="utf-8").write("")
        self.log_to_file("Starting resource summarization...")
        self.query_one("#partial-summary-header", Markdown).update("# Partial Summaries")
        self.run_worker(self.process_summary_with_rag, exclusive=True)

    async def process_summary_with_rag(self) -> None:
        cursor = self.conn.cursor()

        for rtype in RESOURCE_TYPES:
            texts = []
            try:
                self.log_to_file(f"Executing query for patient_id={self.fhirId}, resource_type='{rtype}'")
                cursor.execute("""
                    SELECT STRING(resourcetext)
                    FROM PatientVectors
                    WHERE patient_id = ? AND resource_type = ?
                """, [self.fhirId, rtype])

                i = 0
                while True:
                    try:
                        row = cursor.fetchone()
                        if row is None:
                            break
                        val = row[0]

                        if isinstance(val, list):
                            val = val[0] if val else None

                        if val is None or not isinstance(val, str) or not val.strip():
                            self.log_to_file(f"Skipping row {i} for {rtype} — type: {type(val)}")
                            i += 1
                            continue

                        texts.append(val)
                        self.log_to_file(f"Row {i} for {rtype} accepted: {repr(val)[:100]}")
                        i += 1

                    except Exception as row_err:
                        self.log_to_file(f"Row fetch error at index {i} for {rtype}: {type(row_err).__name__}: {row_err}")
                        break

            except Exception as e:
                self.log_to_file(f"Query failed for {rtype}: {type(e).__name__}: {e}")

            self.log_to_file(f"Texts for {rtype}: count = {len(texts)}")
            if not texts:
                continue

            context = "\n".join(texts)
            context_truncated = self.truncate_to_tokens(context, max_tokens=7000)
            summary = await asyncio.to_thread(self.summarize_resource_type, rtype, context_truncated)
            self.partial_summaries[rtype] = summary.strip()

            self.mount(Markdown(f"## {rtype} Summary\n\n{summary.strip()}"))
            self.progress.advance(1)

        if not self.partial_summaries:
            self.log_to_file("No partial summaries were generated.")
            cursor.close()
            return

        all_text = "\n".join(self.partial_summaries.values())
        final_input = self.truncate_to_tokens(all_text, max_tokens=4000)
        self.log_to_file(f"Generating final summary from combined partials...")
        self.mount(Markdown("\n---\n\n# Final Summary"))
        self.final_summary_text = await asyncio.to_thread(self.summarize_final_summary, final_input)
        self.mount(Markdown("\n\n" + self.final_summary_text.strip()))
        self.log_to_file("Final summary displayed.")
        self.progress.advance(1)
        cursor.close()

    def summarize_resource_type(self, rtype: str, text: str) -> str:
        prompt = (
            f"You are a clinical summarization assistant.\n"
            f"Summarize the following {rtype} information for a patient.\n"
            f"- Limit to 5 sentences or 100 words.\n"
            f"- Use natural language.\n"
            f"- Do not repeat codes or identifiers.\n\n"
            f"--- BEGIN DATA ---\n{text}\n--- END DATA ---\n\nSummary:"
        )
        try:
            response = self.client.chat.completions.create(
              model="gpt-4o-mini",
              messages=[{"role": "user", "content": prompt}],
              max_tokens=8000,
              temperature=0.3,
            )
            summary = response.choices[0].message.content.strip()

            words = summary.split()
            if len(words) > 100:
                summary = " ".join(words[:100]) + "..."
            sentences = re.split(r'(?<=[.!?]) +', summary)
            if len(sentences) > 5:
                summary = " ".join(sentences[:5]) + "..."

            return summary

        except Exception as e:
            self.log_to_file(f"OpenAI error during {rtype}: {e}")
            return f"[OpenAI summarization failed for {rtype}]"

    def summarize_final_summary(self, text: str) -> str:
        prompt = (
            "You are a clinical summarization assistant. Given the following section summaries, write a concise 1–2 paragraph overview of the patient's condition."
            f"\n\n{text}\n\nFinal Summary:"
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=7000,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.log_to_file(f"OpenAI error during final summary: {e}")
            return "[OpenAI final summary failed]"

    def truncate_to_tokens(self, text: str, max_tokens: int = 1500) -> str:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])

    def action_toggle_dark(self) -> None:
        return super().action_toggle_dark()

if __name__ == '__main__':
    app = FHIRSummaryAppOpenAI("2")
    app.run()