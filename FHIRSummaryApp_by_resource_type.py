from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Markdown, ProgressBar
from textual.containers import VerticalScroll
from textual.worker import get_current_worker
import lmstudio as lms
import json, decimal, tiktoken, asyncio, sys
import iris
import traceback
from sentence_transformers import SentenceTransformer

RESOURCE_TYPES = [
    "Patient", "Condition", "MedicationRequest", "Observation",
    "Encounter", "Practitioner", "Procedure", "AllergyIntolerance",
    "Immunization", "DiagnosticReport", "DocumentReference", "CarePlan"
]

class FHIRSummaryAppByResource(App):
    BINDINGS = [('d', 'toggle_dark', 'Toggle dark mode')]

    def __init__(self, ptFHIRid, **kwargs):
        super().__init__(**kwargs)
        self.fhirId = ptFHIRid
        self.client = lms.Client()
        self.model = self.client.llm.model("mistral-7b-instruct-v0.3")
        self.embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        self.conn = iris.connect("127.0.0.1", 1972, "DEMO", "_SYSTEM", "ISCDEMO")
        self.partial_summaries = {}
        self.final_summary_text = ""

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

                rows = []
                i = 0
                while True:
                    try:
                        row = cursor.fetchone()
                        if row is None:
                            break
                        val = row[0]

                        if isinstance(val, list):
                            if not val:
                                self.log_to_file(f"Row {i} for {rtype} is an empty list — skipping")
                                i += 1
                                continue
                            self.log_to_file(f"Row {i} for {rtype} is a list — using first element: {repr(val[0])[:100]}")
                            val = val[0]

                        if val is None or isinstance(val, (int, float, decimal.Decimal)):
                            self.log_to_file(f"Row {i} for {rtype} is non-string: {type(val)} — skipping")
                            i += 1
                            continue

                        if not isinstance(val, str):
                            self.log_to_file(f"Row {i} for {rtype} has unexpected type: {type(val)} — skipping")
                            i += 1
                            continue

                        if not val.strip():
                            self.log_to_file(f"Row {i} for {rtype} is blank — skipping")
                            i += 1
                            continue

                        val.encode("utf-8")
                        texts.append(val)
                        self.log_to_file(f"Row {i} for {rtype} accepted: {repr(val)[:100]}")
                        i += 1

                    except Exception as row_err:
                        self.log_to_file(f"Row fetch error at index {i} for {rtype}: {type(row_err).__name__}: {row_err}")
                        break

            except Exception as e:
                self.log_to_file(f"fetch_rows_for_resource_type failed for {rtype}: {type(e).__name__}: {e}")

            self.log_to_file(f"Texts for {rtype}: count = {len(texts)}")
            if not texts:
                self.log_to_file(f"No texts found for {rtype}, skipping summarization.")
                continue

            context = "\n".join(texts)
            context_truncated = self.truncate_to_tokens(context, max_tokens=1500)
            summary = await asyncio.to_thread(self.summarize_resource_type, rtype, context_truncated)
            self.partial_summaries[rtype] = summary.strip()

            self.mount(Markdown(f"## {rtype} Summary\n\n{summary.strip()}"))
            self.progress.advance(1)

        if not self.partial_summaries:
            self.log_to_file("No partial summaries were generated. Skipping final summary.")
            cursor.close()
            return

        all_text = "\n".join(self.partial_summaries.values())
        final_input = self.truncate_to_tokens(all_text, max_tokens=2000)
        self.log_to_file(f"Partial summaries combined for final summary:\n{final_input[:1000]}...")
        self.mount(Markdown("\n---\n\n# Final Summary"))
        self.final_summary_text = await asyncio.to_thread(self.summarize_final_summary, final_input)
        self.mount(Markdown("\n\n" + self.final_summary_text.strip()))
        self.log_to_file("Final summary updated to display.")
        self.progress.advance(1)
        cursor.close()

    def summarize_resource_type(self, rtype: str, text: str) -> str:
        prompt = (
            f"You are a clinical summarization AI. Summarize the following {rtype} information for a patient in no more than 5 sentences or 100 words."
            f"\n\n{text}\n\nSummary of {rtype}:"
        )
        self.log_to_file(f"🔍 Summarizing {rtype} via LLM...")
        try:
            return self.model.complete(prompt).content
        except AssertionError as e:
            self.log_to_file(f"LLM connection error during {rtype} summarization: {str(e)}")
            return f"[ERROR: LLM connection failed for {rtype}]"

    def summarize_final_summary(self, text: str) -> str:
        prompt = (
            "You are a clinical summarization AI. Using the following section summaries, create a concise, readable 1–2 paragraph overview of the patient's overall clinical picture."
            f"\n\n{text}\n\nFinal Summary:"
        )
        self.log_to_file("Generating final summary via LLM...")
        try:
            return self.model.complete(prompt).content
        except AssertionError as e:
            self.log_to_file(f"LLM connection error during final summary: {str(e)}")
            return "[ERROR: LLM connection failed for final summary]"

    def truncate_to_tokens(self, text: str, max_tokens: int = 1500) -> str:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])

    def action_toggle_dark(self) -> None:
        return super().action_toggle_dark()

if __name__ == '__main__':
    app = FHIRSummaryAppByResource("2")
    app.run()
