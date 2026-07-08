from pathlib import Path

from app.models import RetrievedChunk

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
REVIEW_TEMPLATE_PATH = PROMPTS_DIR / "review.txt"
REVIEW_WITH_RAG_TEMPLATE_PATH = PROMPTS_DIR / "review_with_rag.txt"

DEFAULT_REVIEW_TEMPLATE = """
You are a Principal Software Engineer reviewing a backend system design.

Review the design below. Return a JSON object with:
- confidence: your confidence in the assessment (0.0 to 1.0)
- summary: brief overall assessment including key strengths
- needs_human_review: true if a human architect should review before implementation
- findings: a list of concerns, one per relevant category. Each finding must include:
  - category: reliability, scalability, security, observability, api_design,
    data_consistency, or operational
  - priority: low, medium, high, or critical
  - summary: specific issue, risk, or gap for that category
  - citations: always use an empty list []

Include a finding for every category where you identify an issue. Do not collapse
multiple categories into a single finding.

Design document:
{design_doc}
"""

DEFAULT_RAG_TEMPLATE = """
You are a Principal Software Engineer reviewing a backend system design.

Review the design below. Use the relevant engineering guidance to ground your review.
Reference specific guidance documents when applicable.

Return a JSON object with:
- confidence: your confidence in the assessment (0.0 to 1.0)
- summary: brief overall assessment including key strengths
- needs_human_review: true if a human architect should review before implementation
- findings: a list of concerns, one per relevant category. Each finding must include:
  - category: reliability, scalability, security, observability, api_design,
    data_consistency, or operational
  - priority: low, medium, high, or critical
  - summary: specific issue, risk, or gap for that category
  - citations: list of guidance sources that support this finding. Each citation must
    include source_file, title, page_number, and chunk_number exactly as shown in the
    guidance sections below. Use one or more citations when the finding is grounded in
    retrieved guidance; use an empty list only when based on general engineering knowledge.

Include a finding for every category where you identify an issue. Do not collapse
multiple categories into a single finding. Prefer citing specific guidance with page
and chunk numbers so retrieval is explainable.

Relevant engineering guidance:
{guidance_sections}

Design document:
{design_doc}
"""


class PromptBuilder:
    def __init__(
        self,
        *,
        review_template_path: Path = REVIEW_TEMPLATE_PATH,
        rag_template_path: Path = REVIEW_WITH_RAG_TEMPLATE_PATH,
    ) -> None:
        self.review_template = self._load_template(
            review_template_path, DEFAULT_REVIEW_TEMPLATE
        )
        self.rag_template = self._load_template(
            rag_template_path, DEFAULT_RAG_TEMPLATE
        )

    def build(
        self,
        design_doc: str,
        guidance_chunks: list[RetrievedChunk] | None = None,
    ) -> str:
        if guidance_chunks:
            return self.build_with_retrieval(design_doc, guidance_chunks)
        return self.build_without_retrieval(design_doc)

    def build_without_retrieval(self, design_doc: str) -> str:
        return self.review_template.format(design_doc=design_doc.strip())

    def build_with_retrieval(
        self,
        design_doc: str,
        guidance_chunks: list[RetrievedChunk],
    ) -> str:
        guidance_sections = self._format_guidance(guidance_chunks)
        return self.rag_template.format(
            guidance_sections=guidance_sections,
            design_doc=design_doc.strip(),
        )

    def _format_guidance(self, chunks: list[RetrievedChunk]) -> str:
        sections: list[str] = []
        for chunk in chunks:
            label = chunk.title or chunk.source_file.replace("_", " ").replace(".md", "")
            sections.append(
                f"-------------\n"
                f"source_file: {chunk.source_file}\n"
                f"title: {label}\n"
                f"page_number: {chunk.page_number}\n"
                f"chunk_number: {chunk.chunk_number}\n"
                f"-------------\n"
                f"{chunk.content.strip()}\n"
            )
        return "\n".join(sections)

    @staticmethod
    def _load_template(path: Path, fallback: str) -> str:
        if path.exists():
            return path.read_text(encoding="utf-8")
        return fallback.strip()
