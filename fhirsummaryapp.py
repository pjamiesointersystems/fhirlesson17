from typing import List, Dict
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Markdown, ProgressBar
from textual.worker import get_current_worker
import lmstudio as lms
import json, decimal
import sys
import tiktoken
import asyncio
from fhirpathpy import evaluate as fhirpath
from getSearchPatients import get_everything_for_patient

RESOURCE_TYPES = ["Patient", "Condition", "MedicationRequest", "Observation", "Encounter", "Practitioner"]

class FHIRSummaryApp(App):
    BINDINGS = [('d', 'toggle_dark', 'Toggle dark mode')]

    def __init__(self, ptFHIRid, **kwargs):
        super().__init__(**kwargs)
        self.fhirId = ptFHIRid
        self.client = lms.Client()
        self.model = self.client.llm.model("llama-3.2-3b-instruct")
        self.final_summaries: Dict[str, str] = {}
        self.final_summary_text: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with VerticalScroll():
            yield Markdown("# FHIR Resource Summaries", id="title")
            self.progress = ProgressBar(total=len(RESOURCE_TYPES) + 1, id="progress-bar")
            yield self.progress
            for rtype in RESOURCE_TYPES:
                yield Markdown(f"## {rtype}", id=f"{rtype}-summary")
            yield Markdown("## Final Patient Summary", id="final-summary")

    async def on_mount(self) -> None:
        self.query_one("#title", Markdown).update("Summarizing FHIR Resources by Type...")
        self.run_worker(self.process_summaries, exclusive=True)

    async def process_summaries(self) -> None:
        bundle = self.get_patient_bundle(self.fhirId)
        for rtype in RESOURCE_TYPES:
            resources = self.extract_resources(bundle, rtype)
            if not resources:
                summary_text = f"_No {rtype} resources found._"
            else:
                json_text = json.dumps(resources, indent=2, default=self.make_json_safe)
                chunked = self.truncate_to_tokens(json_text, max_tokens=1500)
                summary_text = await asyncio.to_thread(self.summarize_resource_type, chunked, rtype)
            self.final_summaries[rtype] = summary_text
            self.query_one(f"#{rtype}-summary", Markdown).update(f"### {rtype} Summary\n{summary_text}")
            self.progress.advance(1)
            await asyncio.sleep(0.1)

        # Generate a final summary from all individual summaries
        summary_texts = "\n\n".join(
            f"{rtype}:\n{summary}" for rtype, summary in self.final_summaries.items() if "No" not in summary
        )
        truncated = self.truncate_to_tokens(summary_texts, max_tokens=1500)
        self.final_summary_text = await asyncio.to_thread(self.summarize_final_summary, truncated)
        self.query_one("#final-summary", Markdown).update(self.final_summary_text)
        self.progress.advance(1)

    def summarize_final_summary(self, text: str) -> str:
        prompt = (
            "You are a clinical summarization AI.\n"
            "Using the following individual FHIR resource summaries, produce a concise overall summary of the patient's status in 1–2 paragraphs.\n"
            "Avoid listing resource types or repeating section headings. Be cohesive and medically informative.\n\n"
            f"{text}\n\nFinal Summary:"
        )
        print("🔄 Summarizing final patient summary...")
        return self.model.complete(prompt).content

    def extract_resources(self, bundle: list, resource_type: str) -> list:
        return fhirpath(bundle, f"where(resourceType = '{resource_type}')")

    def summarize_resource_type(self, text: str, rtype: str) -> str:
        prompt = (
            f"You are a clinical summarization AI.\n"
            f"Summarize the key information from the following FHIR {rtype} resources.\n"
            f"Be concise and clear. Do not explain your reasoning.\n\nFHIR {rtype} Data:\n{text}\n\nSummary:"
        )
        print(f"Summarizing {rtype}...")
        return self.model.complete(prompt).content

    def get_patient_bundle(self, patfhirid: str) -> list:
        return get_everything_for_patient(patfhirid)
    
    def get_patient_bundles(self, patfhirid: str) -> list:
        return get_everything_for_patient(patfhirid)

    def truncate_to_tokens(self, text: str, max_tokens: int = 1500) -> str:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        return enc.decode(tokens[:max_tokens])

    def make_json_safe(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def action_toggle_dark(self) -> None:
        return super().action_toggle_dark()


if __name__ == '__main__':
    print(sys.executable)
    app = FHIRSummaryApp("2")
    app.run()

