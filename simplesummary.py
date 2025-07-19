from typing import List, Dict
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.widgets import Header, Footer, Markdown, Button, Input, Select
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fhirId = ""
        self.selected_resource = "Patient"
        self.client = lms.Client()
        self.model = self.client.llm.model("mistral-7b-instruct-v0.3")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield Markdown("# FHIR Resource Summaries", id="title")
        yield Input(placeholder="Enter Patient FHIR ID", id="fhir-id", name="fhir-id")
        yield Select(options=[(rtype, rtype) for rtype in RESOURCE_TYPES], id="resource-select", name="resource-select")
        yield Button(label="Summarize", id="summarize-button", variant="default")
        with VerticalScroll():
            yield Markdown("", id="resource-summary")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "summarize-button":
            input_widget = self.query_one("#fhir-id", Input)
            select_widget = self.query_one("#resource-select", Select)
            self.fhirId = input_widget.value.strip()
            self.selected_resource = select_widget.value
            self.query_one("#title", Markdown).update(f"Summarizing {self.selected_resource} resources for Patient ID {self.fhirId}...")
            self.query_one("#resource-summary", Markdown).update("")
            self.run_worker(self.process_summary, exclusive=True)

    async def process_summary(self) -> None:
        bundle = self.get_patient_bundle(self.fhirId)
        rtype = self.selected_resource
        resources = self.extract_resources(bundle, rtype)
        if not resources:
            summary_text = f"_No {rtype} resources found._"
        else:
            json_text = json.dumps(resources, indent=2, default=self.make_json_safe)
            chunked = self.truncate_to_tokens(json_text, max_tokens=1500)
            summary_text = await asyncio.to_thread(self.summarize_resource_type, chunked, rtype)
        self.query_one("#resource-summary", Markdown).update(f"### {rtype} Summary\n{summary_text}")

    def extract_resources(self, bundle: list, resource_type: str) -> list:
        return fhirpath(bundle, f"where(resourceType = '{resource_type}')")

    def summarize_resource_type(self, text: str, rtype: str) -> str:
        prompt = (
         f"[INST] You are a clinical summarization AI. "
         f"Summarize the key information from the following FHIR {rtype} resources. "
        f"Be concise and clear. Attempt to give health recommendations where possible.\n\n"
        f"{text}\n\n[/INST]"
        )
        print(f"Summarizing {rtype}...")
        return self.model.complete(prompt).content.strip()

    def get_patient_bundle(self, patfhirid: str) -> list:
        return get_everything_for_patient(patfhirid)

    def truncate_to_tokens(self, text: str, max_tokens: int = 1000) -> str:
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
    app = FHIRSummaryApp()
    app.run()
