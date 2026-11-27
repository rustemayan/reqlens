"""Структуры данных аудита."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .criteria import SEVERITY_RANK
from .document import Evidence


@dataclass
class Requirement:
    id: str               # REQ-01 — идентификатор в отчёте
    doc_id: str           # Ф-1 / «п. 3» / "" если номера в документе нет
    section: str
    line_start: int
    line_end: int
    quote: str            # точная цитата из документа
    summary: str          # краткая формулировка
    kind: str = "other"
    page: Optional[int] = None
    assessment: str = ""  # общая оценка модели

    @property
    def lines_label(self) -> str:
        return str(self.line_start) if self.line_start == self.line_end else f"{self.line_start}–{self.line_end}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Question:
    to: str
    text: str


@dataclass
class Finding:
    uid: str                      # временный id (до финальной нумерации)
    scope: str                    # requirement | set
    criterion: str
    severity: str
    title: str
    evidence: List[Evidence]
    explanation: str
    recommendation: str
    question: Optional[Question] = None
    req_ids: List[str] = field(default_factory=list)   # первый — основной
    test_idea: str = ""
    source: str = "llm"           # llm | rule
    rule_id: str = ""
    stage: str = ""
    id: str = ""                  # D-001 — финальный id в отчёте
    merged_from: List[str] = field(default_factory=list)
    review_note: str = ""

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 9)

    @property
    def first_line(self) -> int:
        return min((e.line_start for e in self.evidence), default=10 ** 6)

    @property
    def primary_req(self) -> str:
        return self.req_ids[0] if self.req_ids else ""

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "scope": self.scope,
            "criterion": self.criterion,
            "severity": self.severity,
            "title": self.title,
            "evidence": [e.to_dict() for e in self.evidence],
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "question": asdict(self.question) if self.question else None,
            "req_ids": self.req_ids,
            "test_idea": self.test_idea,
            "source": self.source,
            "rule_id": self.rule_id,
            "merged_from": self.merged_from,
        }
        return d


@dataclass
class GlossaryEntry:
    concept: str
    variants: List[Evidence]
    recommendation: str


@dataclass
class TraceRow:
    source: str        # REQ-xx / цель
    title: str
    ui: str
    api: str
    data: str
    acceptance: str
    status: str        # covered | partial | missing
    note: str = ""


@dataclass
class MissingScenario:
    scenario: str
    why: str
    req_ids: List[str]
    given: str = ""
    when: str = ""
    then: str = ""


@dataclass
class Verdict:
    development: str        # blocked | conditional | ready
    testing: str
    development_reason: str
    testing_reason: str
    summary: str = ""


@dataclass
class Priority:
    finding_ids: List[str]
    action: str


@dataclass
class StageLog:
    name: str
    seconds: float
    status: str
    detail: str = ""


@dataclass
class AuditResult:
    document: Any
    requirements: List[Requirement]
    findings: List[Finding]
    glossary: List[GlossaryEntry] = field(default_factory=list)
    trace: List[TraceRow] = field(default_factory=list)
    missing: List[MissingScenario] = field(default_factory=list)
    priorities: List[Priority] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    stages: List[StageLog] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def findings_for(self, req_id: str, primary_only: bool = False) -> List[Finding]:
        out = []
        for f in self.findings:
            if f.scope != "requirement":
                continue
            if primary_only and f.primary_req != req_id:
                continue
            if req_id in f.req_ids:
                out.append(f)
        return out

    def set_findings(self, criterion: Optional[str] = None) -> List[Finding]:
        return [f for f in self.findings if f.scope == "set" and (criterion is None or f.criterion == criterion)]

    def by_id(self) -> Dict[str, Finding]:
        return {f.id: f for f in self.findings}
