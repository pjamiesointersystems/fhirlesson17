from typing import List
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Markdown, ProgressBar
from textual.worker import Worker, get_current_worker
import lmstudio as lms
import json, decimal
import sys
import tiktoken
import asyncio
from rich.console import Console
from getSearchPatients import get_everything_for_patient


class FHIRApp(App):
    BINDINGS = [('d', 'toggle_dark', 'Toggle dark mode')]

    def __init__(self, ptFHIRid, **kwargs):
        super().__init__(**kwargs)
        self.fhirId = ptFHIRid
        self.client = lms.Client()
        #self.model = self.client.llm.model("deepseek-r1-distill-qwen-7b")
        self.model = self.client.llm.model("llama-3.2-3b-instruct")

        self.batch_summaries: List[str] = []
        self.partial_summaries: List[str] = []
        self.final_summary: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with VerticalScroll():
            yield Markdown("# FHIR Batch Summaries", id="batch-title")
            self.progress = ProgressBar(total=10, id="progress-bar")
            yield self.progress
            for i in range(10):
                yield Markdown(f"### Batch {i+1}\n_Initializing..._", id=f"batch-{i+1}")
            yield Markdown("## Partial Summaries (Batches 1-5):", id="partial-title1")
            yield Markdown("...", id="partial-1")
            yield Markdown("## Partial Summaries (Batches 6-10):", id="partial-title2")
            yield Markdown("...", id="partial-2")
            yield Markdown("## Final Summary:", id="final-title")
            yield Markdown("...", id="final-summary")

    async def on_mount(self) -> None:
            self.query_one("#final-summary", Markdown).update("_Working... please wait._")
            self.run_worker(self.process_summaries, exclusive=True)

    async def process_summaries(self) -> None:
        bundle = self.get_patient_bundle(self.fhirId)
        text = "\n".join(json.dumps(item, indent=2, default=self.make_json_safe) for item in bundle)
        chunks = self.chunk_text_tokenwise(text, max_tokens=1500)[:10]

        self.batch_summaries = []
        for i, chunk in enumerate(chunks):
            self.query_one(f"#batch-{i+1}", Markdown).update("_Loading batch summary..._")
            await asyncio.sleep(0.1)
            summary = await asyncio.to_thread(self.summarize_chunk, chunk, i)
            self.batch_summaries.append(summary)
            self.query_one(f"#batch-{i+1}", Markdown).update(f"### Batch Summary {i+1} \n {summary}")
            self.progress.advance(1)
            await asyncio.sleep(0.1)

        self.final_summary = await asyncio.to_thread(self.merge_summaries, self.batch_summaries)
        if len(self.partial_summaries) >= 2:
            self.query_one("#partial-1", Markdown).update(self.partial_summaries[0].replace("**", "**").replace("- **", "- **"))
            self.query_one("#partial-2", Markdown).update(self.partial_summaries[1].replace("**", "**").replace("- **", "- **"))
        else:
            self.query_one("#partial-1", Markdown).update("_Partial summary unavailable._")
            self.query_one("#partial-2", Markdown).update("_Partial summary unavailable._")
        self.query_one("#final-summary", Markdown).update(self.final_summary.replace("**", "**").replace("- **", "- **"))

    def get_patient_bundle(self, patfhirid: str) -> list:
        return get_everything_for_patient(patfhirid)

    def chunk_text_tokenwise(self, text: str, max_tokens: int = 1000) -> List[str]:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end
        return chunks

    def truncate_to_tokens(self, text: str, max_tokens: int = 700) -> str:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        clipped_tokens = tokens[:max_tokens]
        return enc.decode(clipped_tokens)

    def summarize_chunk(self, chunk: str, chunk_index: int) -> str:
        prompt = (
            f"You are a clinical decision support AI.\n"
            f"Below is a partial FHIR bundle (chunk {chunk_index+1}).\n"
            f"Summarize relevant patient data: diagnoses, medications, labs, key observations.\n"
            f"Do not explain your reasoning or include steps taken to arrive at the summary.\n\n"
            f"FHIR Data Chunk:\n{chunk}\n\nSummary:"
        )
        print(f"--- Summarizing chunk {chunk_index+1} ---")
        return self.model.complete(prompt).content

    def merge_summaries(self, summaries: List[str]) -> str:
        log = Console().log

        print(f"🧠 Merging {len(summaries)} partial summaries in batches...")
        batch_summaries = []
        for i in range(0, len(summaries), 4):
          batch = summaries[i:i+4]
          clipped = [self.truncate_to_tokens(s, max_tokens=700) for s in batch]
          batch_text = "\n\n".join(clipped)
          log(f"Batch {i//4 + 1}: {len(batch_text)} characters")
          batch_prompt = (
            "You are a healthcare AI summarization assistant.\n"
            "Given the following partial summaries, combine them into a coherent intermediate summary.\n\n"
            f"{batch_text}\n\nIntermediate Summary:"
         )
          try:
            result = self.model.complete(batch_prompt)
            summary = result.content or "_LLM returned no intermediate summary._"
            batch_summaries.append(summary)
            log(f"Intermediate summary {i//4 + 1} complete.")
          except Exception as e:
            log(f"Failed to generate intermediate summary: {e}")
            batch_summaries.append("_LLM failed to summarize batch._")

        self.partial_summaries = batch_summaries[:2]

        clipped = [self.truncate_to_tokens(s, max_tokens=450) for s in self.partial_summaries]
        final_text = "\n\n".join(clipped)

        final_prompt = f"""
           You are a clinical summarization AI.
           Using the following intermediate summaries, produce a concise final summary of the patient's condition.
           Do not explain your reasoning or include steps taken to arrive at the summary.

           {final_text}

           Final Patient Summary:
           """

        try:
          final_response = self.model.complete(final_prompt)
          final_result = final_response.content or "_LLM returned no final summary._"
          log(" Final summary generated successfully.")
          return final_result
        except Exception as e:
          log(f"Final summary generation failed: {e}")
          return "Final summary could not be generated due to an error."


    def make_json_safe(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def action_toggle_dark(self) -> None:
        return super().action_toggle_dark()


if __name__ == '__main__':
    print(sys.executable)
    app = FHIRApp("2")
    app.run()
