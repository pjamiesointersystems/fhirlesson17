from regex import R
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Button, Markdown, Static
from textual.containers import VerticalScroll, Vertical
import iris
import tiktoken
from sentence_transformers import SentenceTransformer
import lmstudio as lms
import asyncio
from typing import List, Tuple

VECTOR_TABLE = "PatientVectors"  # or "PatientSummaryVectors"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_DIM = 768
TOP_K = 4

def embed_text(model, text: str) -> List[float]:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    # optionally truncate tokens here
    vec = model.encode(text).tolist()
    return vec

class FHIRRAGChatApp(App):
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # LLM client
        self.client = lms.Client()
        self.llm = self.client.llm.model("mistral-7b-instruct-v0.3")
        # Embedding model
        self.embedder = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        # IRIS connection
        self.conn = iris.connect("127.0.0.1", 1972, "DEMO", "_SYSTEM", "ISCDEMO")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with VerticalScroll(id="chat-log"):
        # Input row
         with Vertical():
            yield Markdown("# RAG Chat\n\nEnter a FHIR Id and a question, then press Ask.")
            yield Input(placeholder="Enter FHIR Patient Id", id="fhir-id")
            yield Input(placeholder="Enter your question here", id="query")
            yield Button("Ask", id="ask-btn", variant="primary")
            

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        # Guard so we only handle our “Ask” button
        if event.button.id != "ask-btn":
            return

        # Grab the inputs
        fhir_id = self.query_one("#fhir-id", Input).value.strip()
        query   = self.query_one("#query", Input).value.strip()
        if not fhir_id or not query:
            await self.post_message(Static("🚨 Please fill both fields."))
            return

        # Reference to the chat log area
        chat = self.query_one("#chat-log", VerticalScroll)

        # This line must live inside an async function:
        answer, first, last = await asyncio.to_thread(self.run_rag, fhir_id, query)

        # Echo the user and bot messages
        chat.mount(Markdown(f"**User (Patient {first} {last}, ID={fhir_id}):** {query}\n"))
        chat.mount(Markdown(f"**Bot (for {first} {last}):** {answer}\n"))

        # Scroll to the bottom
        chat.scroll_end(animate=False)

            
      

    def run_rag(self, fhir_id: str, query: str) -> str:
        # 1) embed the query
        vec = embed_text(self.embedder, query)
        csv = ",".join(f"{x:.8f}" for x in vec)

        # 2) retrieve top-K passages for this patient
        sql = f"""
          SELECT TOP {TOP_K}
            resource_id,
            VECTOR_COSINE(embedding, TO_VECTOR(?,DOUBLE)) AS score
          FROM {VECTOR_TABLE}
          WHERE patient_id = ? AND resource_type = 'Condition'
          ORDER BY score DESC
        """
        cur = self.conn.cursor()
        cur.execute(sql, [csv, fhir_id])
        id_sim_pairs = cur.fetchall()

        if not id_sim_pairs:
            return "No matching data found for that patient."
        
        cur = self.conn.cursor()
        


       # Step 2: for each ID, fetch the text separately
        results = []
        text_sql = "SELECT CAST(resourcetext AS VARCHAR(4000)), patient_lastname, patient_firstname FROM PatientVectors WHERE resource_id = ?"
        for rid, sim in id_sim_pairs:
            cur.execute(text_sql, [rid])
            (text, ptLastName, ptFirstName) = cur.fetchone()
            results.append((rid, text, ptLastName, ptFirstName,sim))

        # 3) assemble context
        _, _, ptLastName, ptFirstName, _ = results[0]
        passages = [text for (_rid, text, ptLastName, ptFirstName, _sim) in results]
        # then join with double-newlines
        context = "\n\n".join(passages)

        # 4) build RAG prompt
        prompt = (
            "You are a clinical assistant. Using the following extracted patient data, "
            "answer the user’s question precisely and concisely.\n\n"
            "--- BEGIN CONTEXT ---\n"
            f"{context}\n"
            "--- END CONTEXT ---\n\n"
            f"Question: {query}\nAnswer:"
        )

        # 5) call LLM
        resp = self.llm.complete(prompt)
        answer = resp.content.strip()
        return answer, ptFirstName, ptLastName

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

if __name__ == "__main__":
    FHIRRAGChatApp().run()
