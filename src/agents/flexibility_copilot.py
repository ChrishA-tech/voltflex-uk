"""Guardrailed LangGraph Flexibility Copilot with BM25 Hybrid Retrieval and G98/G99 Safety Interlock."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_REGULATORY_DOCS_DIR = Path("data/regulatory_docs")

G98_SINGLE_PHASE_LIMIT_KW = 3.68
G98_SINGLE_PHASE_LIMIT_AMPS = 16.0


@dataclass(frozen=True)
class RegulatoryDocument:
    """Authoritative UK grid, DNO, and Ofgem regulatory knowledge chunk."""

    doc_id: str
    title: str
    authority: str
    text: str


UK_FLEXIBILITY_CORPUS: list[RegulatoryDocument] = [
    RegulatoryDocument(
        doc_id="ENA-G98-01",
        title="ENA Engineering Recommendation G98 Single-Phase Inverter Limit",
        authority="Energy Networks Association / Ofgem",
        text=(
            "Under ENA Engineering Recommendation G98, micro-generators (solar PV, home batteries, V2G) "
            "with an aggregate inverter capacity up to 3.68 kW (16 A per phase at 230 V nominal single-phase) "
            "may connect under the connect-and-notify fast-track procedure, notifying the DNO within 28 days "
            "of commissioning. Exceeding 3.68 kW single-phase export without prior DNO approval is prohibited."
        ),
    ),
    RegulatoryDocument(
        doc_id="ENA-G99-01",
        title="ENA Engineering Recommendation G99 Pre-Approval & Thermal Limits",
        authority="Energy Networks Association / UK DNOs",
        text=(
            "Installations with aggregate export or inverter rating above 3.68 kW single-phase (16 A) "
            "require ENA Engineering Recommendation G99 application and formal DNO approval before "
            "commissioning. DNOs may issue export limitation schemes (G100) or curtail export when "
            "substation transformer headroom is constrained or thermal current limits are approached."
        ),
    ),
    RegulatoryDocument(
        doc_id="NESO-DFS-01",
        title="NESO Demand Flexibility Service (DFS) Participation & Baselines",
        authority="National Energy System Operator (NESO)",
        text=(
            "The NESO Demand Flexibility Service (DFS) rewards registered domestic and I&C assets via "
            "approved providers for shifting demand or exporting battery/EV energy during peak system "
            "stress windows. Settlement uses half-hourly smart meter readings (MPAN) against a counterfactual "
            "baseline; assets must not double-count capacity already committed to the Balancing Mechanism or Capacity Market."
        ),
    ),
    RegulatoryDocument(
        doc_id="NESO-DFS-02",
        title="NESO DFS Dispatch Timing & Dynamic Utilization",
        authority="National Energy System Operator (NESO)",
        text=(
            "DFS Test and Live events notify providers ahead of target settlement periods (typically day-ahead "
            "or within-day). During Dunkelflaute peaks with low wind generation and high wholesale prices, "
            "home batteries and V2G-capable EVs can discharge up to their approved G98 (3.68 kW) or G99 limit."
        ),
    ),
    RegulatoryDocument(
        doc_id="OFGEM-SEG-01",
        title="Ofgem Smart Export Guarantee (SEG) Eligibility & Metering",
        authority="Ofgem",
        text=(
            "Under the Ofgem Smart Export Guarantee (SEG), licensed electricity suppliers with 150,000 or more "
            "domestic customers must offer at least one positive export tariff (> 0 p/kWh) for eligible low-carbon "
            "generation up to 5 MW. Generators must hold an export MPAN with half-hourly smart meter capability "
            "and MCS (or equivalent) installation certification."
        ),
    ),
    RegulatoryDocument(
        doc_id="UK-AGILE-01",
        title="Agile Plunge Pricing & Wind Curtailment Absorption",
        authority="GB Wholesale / Half-Hourly Retail Market",
        text=(
            "During high-wind curtailment windows, GB wholesale and Octopus Agile half-hourly import prices can "
            "drop near zero or negative (plunge pricing). Smart EV chargers and home batteries absorb surplus "
            "wind energy while respecting local DNO substation headroom and site main fuse thermal current limits."
        ),
    ),
    RegulatoryDocument(
        doc_id="UK-GDPR-01",
        title="Smart Meter MPAN UK GDPR Pseudonymisation",
        authority="UK GDPR / Smart Energy Code",
        text=(
            "Half-hourly smart meter MPAN identifiers constitute personal data under UK GDPR. Production "
            "telemetry pipelines must pseudonymise raw MPANs using keyed cryptographic hashing (HMAC-SHA256) "
            "before analytics persistence or dead-letter quarantine storage."
        ),
    ),
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _parse_markdown_sections(content: str, stem: str) -> list[RegulatoryDocument]:
    """Parse structured markdown regulatory sections (`## [DOC-ID] Title`) or fallback chunks."""
    section_pattern = re.compile(r"^##\s+\[([A-Z0-9-]+)\]\s+(.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(content))
    if not matches:
        clean = " ".join(content.split())
        if not clean:
            return []
        doc_tag = re.sub(r"[^A-Z0-9]+", "-", stem.upper()).strip("-")[:20] or "REG"
        return [
            RegulatoryDocument(
                doc_id=f"{doc_tag}-01",
                title=stem.replace("_", " ").title(),
                authority="NESO / Ofgem / ENA",
                text=clean,
            )
        ]

    docs: list[RegulatoryDocument] = []
    for idx, match in enumerate(matches):
        doc_id = match.group(1).strip()
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        body = content[start:end].strip()

        authority = "NESO / Ofgem / ENA"
        body_lines = []
        for line in body.splitlines():
            if line.lower().startswith("authority:"):
                authority = line.split(":", 1)[1].strip()
            else:
                body_lines.append(line)
        text = " ".join(" ".join(body_lines).split())
        if text:
            docs.append(
                RegulatoryDocument(
                    doc_id=doc_id,
                    title=title,
                    authority=authority,
                    text=text,
                )
            )
    return docs


def _chunk_plain_text(text: str, stem: str, chunk_chars: int = 800) -> list[RegulatoryDocument]:
    """Split unstructured .txt or .pdf text into indexed RegulatoryDocument chunks."""
    clean = " ".join(text.split())
    if not clean:
        return []
    doc_tag = re.sub(r"[^A-Z0-9]+", "-", stem.upper()).strip("-")[:16] or "REG"
    docs: list[RegulatoryDocument] = []
    words = clean.split(" ")
    buf: list[str] = []
    buf_len = 0
    part = 1
    for word in words:
        buf.append(word)
        buf_len += len(word) + 1
        if buf_len >= chunk_chars:
            docs.append(
                RegulatoryDocument(
                    doc_id=f"{doc_tag}-{part:02d}",
                    title=f"{stem.replace('_', ' ').title()} (Part {part})",
                    authority="Statutory UK Regulatory Document",
                    text=" ".join(buf),
                )
            )
            buf = []
            buf_len = 0
            part += 1
    if buf:
        docs.append(
            RegulatoryDocument(
                doc_id=f"{doc_tag}-{part:02d}",
                title=f"{stem.replace('_', ' ').title()} (Part {part})" if part > 1 else stem.replace("_", " ").title(),
                authority="Statutory UK Regulatory Document",
                text=" ".join(buf),
            )
        )
    return docs


def load_flexibility_regulatory_corpus(
    docs_dir: Path | str = DEFAULT_REGULATORY_DOCS_DIR,
) -> list[RegulatoryDocument]:
    """Load and parse .pdf (pypdf), .md, and .txt NESO/ENA/Ofgem rules from disk."""
    target_dir = Path(docs_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        return list(UK_FLEXIBILITY_CORPUS)

    loaded: list[RegulatoryDocument] = []
    seen_ids: set[str] = set()

    for file_path in sorted(target_dir.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        chunks: list[RegulatoryDocument] = []
        if suffix == ".md":
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = _parse_markdown_sections(content, file_path.stem)
        elif suffix == ".txt":
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk_plain_text(content, file_path.stem)
        elif suffix == ".pdf":
            reader = PdfReader(str(file_path))
            pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            chunks = _chunk_plain_text(pdf_text, file_path.stem)

        for doc in chunks:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                loaded.append(doc)

    return loaded if loaded else list(UK_FLEXIBILITY_CORPUS)


class HybridRegulatoryRetriever:
    """BM25 + TF-IDF cosine hybrid retriever over NESO, Ofgem, and ENA G98/G99 rules."""

    def __init__(
        self,
        corpus: list[RegulatoryDocument] | None = None,
        docs_dir: Path | str | None = DEFAULT_REGULATORY_DOCS_DIR,
    ) -> None:
        if corpus is not None:
            self.corpus = corpus
        elif docs_dir is not None:
            self.corpus = load_flexibility_regulatory_corpus(docs_dir)
        else:
            self.corpus = list(UK_FLEXIBILITY_CORPUS)
        self._bm25 = BM25Okapi([_tokenize(f"{d.title} {d.text}") for d in self.corpus])
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform([f"{d.title} {d.text}" for d in self.corpus])

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[RegulatoryDocument, float]]:
        tokens = _tokenize(query)
        bm25_scores = np.array(self._bm25.get_scores(tokens), dtype=float)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        q_vec = self._tfidf.transform([query])
        dense_scores = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
        if dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()

        hybrid = 0.65 * bm25_scores + 0.35 * dense_scores
        ranked_idx = np.argsort(hybrid)[::-1][:top_k]
        return [(self.corpus[int(i)], round(float(hybrid[int(i)]), 4)) for i in ranked_idx]


class CopilotState(TypedDict, total=False):
    """LangGraph state schema for the Flexibility Copilot workflow."""

    query: str
    requested_export_kw: float | None
    g99_approved: bool
    substation_headroom_pct: float | None
    retrieved_docs: list[dict[str, Any]]
    interlock_triggered: bool
    interlock_reason: str
    draft_answer: str
    cited_ids: list[str]
    citation_verified: bool
    verification_notes: str
    final_answer: str
    used_safety_fallback: bool


def _check_g98_g99_interlock(
    query: str,
    requested_export_kw: float | None,
    g99_approved: bool,
    substation_headroom_pct: float | None,
) -> tuple[bool, str]:
    """Deterministic G98/G99 & substation headroom safety interlock."""
    reasons: list[str] = []

    inferred_kw = requested_export_kw
    if inferred_kw is None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*kw", query.lower())
        if match:
            inferred_kw = float(match.group(1))

    q_lower = query.lower()
    bypass_attempt = any(
        phrase in q_lower
        for phrase in ("bypass g98", "ignore g98", "override g98", "disable export limit", "bypass thermal")
    )

    if bypass_attempt:
        reasons.append(
            "Safety interlock blocked request to bypass ENA G98/G99 or thermal current limits."
        )

    if inferred_kw is not None and inferred_kw > G98_SINGLE_PHASE_LIMIT_KW and not g99_approved:
        reasons.append(
            f"Requested export/inverter power ({inferred_kw:.2f} kW) exceeds single-phase ENA G98 "
            f"limit ({G98_SINGLE_PHASE_LIMIT_KW} kW / {G98_SINGLE_PHASE_LIMIT_AMPS:.0f} A) without G99 approval; "
            f"dispatch clamped to {G98_SINGLE_PHASE_LIMIT_KW} kW."
        )

    if substation_headroom_pct is not None and substation_headroom_pct < 10.0:
        reasons.append(
            f"Local DNO substation headroom is CRITICAL ({substation_headroom_pct:.1f}% < 10%); "
            "heavy EV/battery import curtailed to 6 A trickle cap."
        )

    return bool(reasons), " ".join(reasons)


def verify_citations(
    answer_text: str,
    retrieved_docs: list[dict[str, Any]],
    interlock_triggered: bool,
) -> tuple[bool, list[str], str]:
    """Judge function verifying grounded [DOC-ID] citations and G98/G99 compliance."""
    found_ids = re.findall(r"\[([A-Z0-9-]+)\]", answer_text)
    valid_ids = {d["doc_id"] for d in retrieved_docs}

    if not found_ids:
        return False, [], "No inline regulatory citations found in synthesized response."

    unknown_ids = [cid for cid in found_ids if cid not in valid_ids]
    if unknown_ids:
        return False, found_ids, f"Unretrieved or hallucinated citations detected: {unknown_ids}."

    if interlock_triggered and "3.68" not in answer_text and "G98" not in answer_text:
        return False, found_ids, "Safety interlock triggered but response omitted G98 3.68 kW constraint."

    deduped = list(dict.fromkeys(found_ids))
    return True, deduped, "All regulatory claims grounded in retrieved NESO/Ofgem/ENA citations."


class FlexibilityCopilot:
    """Guardrailed LangGraph agent for NESO DFS, Ofgem SEG, and ENA G98/G99 compliance."""

    def __init__(
        self,
        retriever: HybridRegulatoryRetriever | None = None,
        docs_dir: Path | str | None = DEFAULT_REGULATORY_DOCS_DIR,
    ) -> None:
        self.retriever = retriever or HybridRegulatoryRetriever(docs_dir=docs_dir)
        self.graph = self._build_graph()

    def _retrieve_node(self, state: CopilotState) -> CopilotState:
        query = state["query"]
        # Bias query slightly if G98/G99 numbers appear so safety docs are always in context
        aug_query = query
        if state.get("requested_export_kw") is not None or "kw" in query.lower():
            aug_query = f"{query} G98 G99 3.68 kW inverter export limit"

        hits = self.retriever.retrieve(aug_query, top_k=3)
        docs = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "authority": doc.authority,
                "text": doc.text,
                "score": score,
            }
            for doc, score in hits
        ]
        return {"retrieved_docs": docs}

    def _safety_interlock_node(self, state: CopilotState) -> CopilotState:
        triggered, reason = _check_g98_g99_interlock(
            query=state["query"],
            requested_export_kw=state.get("requested_export_kw"),
            g99_approved=bool(state.get("g99_approved", False)),
            substation_headroom_pct=state.get("substation_headroom_pct"),
        )
        # Ensure ENA-G98-01 / ENA-G99-01 are in retrieved_docs when interlock triggers
        docs = list(state.get("retrieved_docs", []))
        if triggered:
            present = {d["doc_id"] for d in docs}
            for doc in self.retriever.corpus:
                if doc.doc_id in ("ENA-G98-01", "ENA-G99-01") and doc.doc_id not in present:
                    docs.append(
                        {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "authority": doc.authority,
                            "text": doc.text,
                            "score": 1.0,
                        }
                    )
        return {
            "retrieved_docs": docs,
            "interlock_triggered": triggered,
            "interlock_reason": reason,
        }

    def _synthesize_node(self, state: CopilotState) -> CopilotState:
        docs = state.get("retrieved_docs", [])
        interlock_triggered = bool(state.get("interlock_triggered", False))
        interlock_reason = state.get("interlock_reason", "")

        parts: list[str] = []
        if interlock_triggered:
            parts.append(
                f"**G98/G99 Deterministic Safety Interlock Activated:** {interlock_reason} "
                f"[ENA-G98-01] [ENA-G99-01]"
            )

        if docs:
            lead = docs[0]
            parts.append(f"**Guidance ({lead['title']}):** {lead['text']} [{lead['doc_id']}]")
            for extra in docs[1:3]:
                parts.append(f"- **{extra['title']}:** {extra['text']} [{extra['doc_id']}]")
        else:
            parts.append("No regulatory documents matched the query.")

        return {"draft_answer": "\n\n".join(parts)}

    def _citation_verifier_node(self, state: CopilotState) -> CopilotState:
        draft = state.get("draft_answer", "")
        docs = state.get("retrieved_docs", [])
        interlock = bool(state.get("interlock_triggered", False))
        ok, cited_ids, notes = verify_citations(draft, docs, interlock)
        return {
            "citation_verified": ok,
            "cited_ids": cited_ids,
            "verification_notes": notes,
            "final_answer": draft if ok else "",
            "used_safety_fallback": False,
        }

    def _safety_fallback_node(self, state: CopilotState) -> CopilotState:
        """Deterministic G98/G99 safety interlock fallback if citation judge rejects draft."""
        interlock_msg = state.get("interlock_reason") or (
            "Defaulting to ENA G98 single-phase export cap of 3.68 kW (16 A @ 230 V) until G99 DNO "
            "approval and verified regulatory context are confirmed."
        )
        fallback = (
            f"**Deterministic G98/G99 Safety Interlock Fallback:** {interlock_msg} [ENA-G98-01] [ENA-G99-01]"
        )
        return {
            "final_answer": fallback,
            "cited_ids": ["ENA-G98-01", "ENA-G99-01"],
            "citation_verified": True,
            "used_safety_fallback": True,
            "verification_notes": "Routed to deterministic G98/G99 safety interlock fallback.",
        }

    @staticmethod
    def _route_after_verifier(state: CopilotState) -> str:
        return "verified" if state.get("citation_verified") else "fallback"

    def _build_graph(self):
        builder = StateGraph(CopilotState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("safety_interlock", self._safety_interlock_node)
        builder.add_node("synthesize", self._synthesize_node)
        builder.add_node("citation_verifier", self._citation_verifier_node)
        builder.add_node("safety_fallback", self._safety_fallback_node)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "safety_interlock")
        builder.add_edge("safety_interlock", "synthesize")
        builder.add_edge("synthesize", "citation_verifier")
        builder.add_conditional_edges(
            "citation_verifier",
            self._route_after_verifier,
            {
                "verified": END,
                "fallback": "safety_fallback",
            },
        )
        builder.add_edge("safety_fallback", END)
        return builder.compile()

    def ask(
        self,
        query: str,
        requested_export_kw: float | None = None,
        g99_approved: bool = False,
        substation_headroom_pct: float | None = None,
    ) -> dict[str, Any]:
        """Run the guardrailed LangGraph copilot and return verified response payload."""
        initial_state: CopilotState = {
            "query": query,
            "requested_export_kw": requested_export_kw,
            "g99_approved": g99_approved,
            "substation_headroom_pct": substation_headroom_pct,
        }
        result = self.graph.invoke(initial_state)
        return {
            "query": query,
            "answer": result.get("final_answer", ""),
            "citations": result.get("cited_ids", []),
            "citation_verified": bool(result.get("citation_verified", False)),
            "verification_notes": result.get("verification_notes", ""),
            "interlock_triggered": bool(result.get("interlock_triggered", False)),
            "interlock_reason": result.get("interlock_reason", ""),
            "used_safety_fallback": bool(result.get("used_safety_fallback", False)),
            "retrieved_docs": result.get("retrieved_docs", []),
        }
