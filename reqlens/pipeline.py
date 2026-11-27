"""Оркестрация прогона: документ → правила → нарезка → проверка требований →
проверка набора → ревью находок → вердикт.

Каждый этап LLM изолирован: сбой одного этапа не роняет прогон, а фиксируется
в журнале и в отчёте. Все находки проходят проверку доказательств: цитаты ищутся
в документе кодом, находки без подтверждённой цитаты отбрасываются.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import prompts, schemas
from .criteria import (
    ADDRESSEES,
    ALL,
    INDIVIDUAL_IDS,
    SET_IDS,
    SEVERITIES,
    SEVERITY_RANK,
)
from .document import Document, Evidence, normalize
from .llm import LLMClient, LLMError, LLMFatalError, LLMTruncated, summarize_records
from .loaders import load_document
from .models import (
    AuditResult,
    Finding,
    GlossaryEntry,
    MissingScenario,
    Priority,
    Question,
    Requirement,
    StageLog,
    TraceRow,
)
from .rules import run_rules
from .segmenter import coverage, display_coverage, find_requirement, from_llm, number, structural_segment
from .utils import agent_version
from .verdict import build_verdict, fallback_priorities

log = logging.getLogger("reqlens")

# соответствие «критерий требования → критерий набора» для находок вне требований
_TO_SET = {
    "unambiguous": "set_comprehensible",
    "conforming": "set_feasible",
    "feasible": "set_feasible",
    "correct": "set_consistent",
    "consistent": "set_consistent",
    "verifiable": "set_validatable",
    "complete": "set_complete",
    "singular": "traceable",
    "necessary": "set_complete",
    "appropriate": "set_feasible",
}
_TO_REQ = {
    "set_comprehensible": "unambiguous",
    "set_feasible": "feasible",
    "set_consistent": "consistent",
    "set_validatable": "verifiable",
    "set_complete": "complete",
    "traceable": "singular",
    "ranked": "necessary",
    "modifiable": "conforming",
}
# в матрицу прослеживаемости попадают источники требований, а не строки паспорта API или примеры ответов
_TRACE_KINDS = {"goal", "functional", "nonfunctional", "integration", "data", "ui", "process"}

_FAMILY = {
    "conforming": "std", "feasible": "std", "set_feasible": "std", "appropriate": "std",
    "consistent": "cons", "set_consistent": "cons", "correct": "cons",
    "complete": "compl", "set_complete": "compl",
    "unambiguous": "amb", "set_comprehensible": "amb",
    "verifiable": "ver", "set_validatable": "ver",
    "modifiable": "mod", "traceable": "trace", "singular": "trace", "ranked": "rank", "necessary": "nec",
}


class Auditor:
    def __init__(
        self,
        llm: Optional[LLMClient],
        batch_size: int = 10,
        review_pass: bool = True,
        progress: Optional[Callable[[str], None]] = None,
    ):
        self.llm = llm
        self.batch_size = max(1, batch_size)
        self.review_pass = review_pass
        self.progress = progress or (lambda msg: None)
        self.stages: List[StageLog] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, int] = {
            "quotes_total": 0,
            "quotes_exact": 0,
            "quotes_fuzzy": 0,
            "quotes_rejected": 0,
            "findings_unverified": 0,
            "merged_auto": 0,
            "merged_review": 0,
            "dropped_review": 0,
            "severity_changed_review": 0,
        }
        self.rejected: List[Tuple[str, str]] = []  # (title, reason)
        self._uid = 0

    # ------------------------------------------------------------- helpers
    def _stage(self, name: str, fn, *args, **kwargs):
        t0 = time.monotonic()
        self._last_detail = ""
        try:
            res = fn(*args, **kwargs)
            self.stages.append(StageLog(name, time.monotonic() - t0, "ok", self._last_detail))
            return res
        except LLMFatalError:
            raise
        except LLMError as exc:
            msg = f"Этап «{name}» не выполнен: {exc}"
            log.warning(msg)
            self.warnings.append(msg)
            self.stages.append(StageLog(name, time.monotonic() - t0, "ошибка", str(exc)[:200]))
            return None

    _last_detail = ""

    def _detail(self, text: str) -> None:
        self._last_detail = text

    def _next_uid(self, prefix: str) -> str:
        self._uid += 1
        return f"{prefix}{self._uid}"

    def _ask(self, stage: str, user: str, schema: dict):
        assert self.llm is not None
        return self.llm.complete_json(stage, prompts.SYSTEM, user, schema)

    # ----------------------------------------------------------------- run
    def run(self, path: Path) -> AuditResult:
        started = datetime.now().astimezone()
        t0 = time.monotonic()

        self.progress(f"[1/7] Чтение документа: {path}")
        doc = load_document(path)
        self.stages.append(StageLog("Чтение документа", time.monotonic() - t0, "ok", f"{doc.fmt}, {doc.line_count} строк"))
        self.progress(f"      {doc.fmt}, {doc.line_count} строк, «{doc.title}»")

        self.progress("[2/7] Статические правила")
        t1 = time.monotonic()
        rule_findings = run_rules(doc)
        self.stats["rules_fired"] = {f.rule_id: 1 for f in rule_findings}
        self.stages.append(StageLog("Статические правила", time.monotonic() - t1, "ok", f"{len(rule_findings)} находок"))
        self.progress(f"      найдено {len(rule_findings)}")

        self.progress("[3/7] Нарезка на требования")
        reqs = self._segment(doc)
        cov = display_coverage(doc, reqs)
        self.progress(f"      {len(reqs)} требований, покрытие текста {cov:.0%}")

        req_findings: List[Finding] = []
        set_findings: List[Finding] = []
        glossary: List[GlossaryEntry] = []
        trace: List[TraceRow] = []
        missing: List[MissingScenario] = []

        if self.llm:
            self.progress("[4/7] Проверка каждого требования")
            res = self._stage("Проверка требований", self._check_requirements, doc, reqs, rule_findings)
            if res:
                req_findings = res

            self.progress("[5/7] Проверка набора: согласованность и терминология")
            res2 = self._stage("Проверка набора: согласованность", self._check_set_consistency, doc, reqs, rule_findings + req_findings)
            if res2:
                set_findings.extend(res2[0])
                glossary = res2[1]
            self.progress("      полнота и прослеживаемость")
            res3 = self._stage("Проверка набора: полнота", self._check_set_completeness, doc, reqs, rule_findings + req_findings + set_findings)
            if res3:
                set_findings.extend(res3[0])
                trace, missing = res3[1], res3[2]
        else:
            self.progress("[4/7] Пропуск этапов LLM (режим --offline)")

        findings = rule_findings + req_findings + set_findings
        findings = self._attach(doc, reqs, findings)
        findings = self._auto_merge(findings)

        review = None
        if self.llm and self.review_pass and findings:
            self.progress(f"[6/7] Ревью находок (скептик): {len(findings)}")
            review = self._stage("Ревью находок", self._review, doc, reqs, findings)
            if review:
                findings = review[0]
        else:
            self.progress("[6/7] Ревью находок пропущено")

        findings = self._merge_subsumed_contradictions(findings)
        findings = self._number(findings, reqs)
        uid_to_id = {}
        for f in findings:
            uid_to_id[f.uid] = f.id
            for m in f.merged_from:
                uid_to_id[m] = f.id

        priorities: List[Priority] = []
        summary = ""
        dev_reason = test_reason = ""
        if review:
            _, raw_prior, summary, dev_reason, test_reason = review
            for p in raw_prior:
                ids = []
                for u in p.get("finding_ids") or []:
                    fid = uid_to_id.get(str(u))
                    if fid and fid not in ids:
                        ids.append(fid)
                action = str(p.get("action") or "").strip()
                if action and ids:
                    priorities.append(Priority(ids, action))
        if not priorities:
            priorities = fallback_priorities(findings)

        verdict = build_verdict(findings, reqs, summary, dev_reason, test_reason)
        self.progress("[7/7] Формирование отчёта")

        llm_stats = summarize_records(self.llm.records) if self.llm else {}
        meta = {
            "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
            "utc_offset": started.strftime("%z"),
            "duration_s": round(time.monotonic() - t0, 1),
            "agent": agent_version(),
            "mode": "LLM + статические правила" if self.llm else "только статические правила (--offline)",
            "provider": self.llm.settings.provider_title if self.llm else "—",
            "model": self.llm.settings.model if self.llm else "—",
            "used_models": list(self.llm.used_models) if self.llm else [],
            "batch_size": self.batch_size,
            "review_pass": bool(self.llm and self.review_pass),
            "llm": llm_stats,
            "coverage": round(cov, 3),
        }
        stats = dict(self.stats)
        stats["requirements"] = len(reqs)
        stats["requirements_numbered"] = len([r for r in reqs if r.doc_id])
        stats["rejected"] = self.rejected
        return AuditResult(
            document=doc,
            requirements=reqs,
            findings=findings,
            glossary=glossary,
            trace=trace,
            missing=missing,
            priorities=priorities,
            verdict=verdict,
            stages=self.stages,
            meta=meta,
            stats=stats,
            warnings=self.warnings,
        )

    # ------------------------------------------------------------ segment
    def _segment(self, doc: Document) -> List[Requirement]:
        t0 = time.monotonic()
        if self.llm:
            try:
                data = self._ask("segment", prompts.segment_prompt(doc), schemas.SEGMENT_SCHEMA)
                reqs, st = from_llm(doc, data if isinstance(data, dict) else {})
                cov = coverage(doc, reqs)
                detail = (f"модель предложила {st['proposed']}, подтверждено цитатой {st['verified']}, "
                          f"по диапазону строк {st['by_range']}, отброшено {st['dropped']}, дублей {st['duplicates']}")
                if reqs and cov < 0.85:
                    # гибрид: добавляем структурные единицы для строк, которые модель пропустила
                    added = 0
                    for s in structural_segment(doc):
                        if any(not (s.line_end < r.line_start or s.line_start > r.line_end) for r in reqs):
                            continue
                        first = (doc.line(s.line_start).text if doc.line(s.line_start) else "").strip()
                        # дополняем только содержательный текст: служебные строки таблиц, примеры JSON
                        # и короткие заглушки модель пропустила намеренно
                        if first.startswith(("|", "{", "[", "curl")) or len(s.quote.split()) < 4:
                            continue
                        reqs.append(s)
                        added += 1
                    if added:
                        reqs = number(reqs)
                        detail += f"; добавлено структурной нарезкой {added} (покрытие моделью было {cov:.0%})"
                        cov = coverage(doc, reqs)
                if len(reqs) >= 3:
                    self.stages.append(StageLog("Нарезка (LLM)", time.monotonic() - t0, "ok", detail))
                    self.stats["segment_dropped"] = st["dropped"]
                    return reqs
                self.warnings.append(f"Нарезка моделью неполная — использована структурная нарезка. {detail}")
            except LLMFatalError:
                raise
            except LLMError as exc:
                self.warnings.append(f"Нарезка моделью не удалась ({exc}) — использована структурная нарезка.")
        reqs = structural_segment(doc)
        self.stages.append(StageLog("Нарезка (структурная)", time.monotonic() - t0, "ok", f"{len(reqs)} требований"))
        return reqs

    # --------------------------------------------------- per-requirement
    def _check_requirements(self, doc: Document, reqs: List[Requirement], rule_findings: List[Finding]) -> List[Finding]:
        by_id = {r.id: r for r in reqs}
        results: Dict[str, dict] = {}
        batches = [reqs[i : i + self.batch_size] for i in range(0, len(reqs), self.batch_size)]
        for n, batch in enumerate(batches, start=1):
            self.progress(f"      пакет {n}/{len(batches)}: {batch[0].id}…{batch[-1].id}")
            self._run_batch(doc, reqs, batch, rule_findings, results)
        missing = [r for r in reqs if r.id not in results]
        if missing and len(missing) <= self.batch_size * 2:
            self.progress(f"      дозапрос пропущенных требований: {len(missing)}")
            for i in range(0, len(missing), self.batch_size):
                self._run_batch(doc, reqs, missing[i : i + self.batch_size], rule_findings, results)
        still = [r.id for r in reqs if r.id not in results]
        if still:
            self.warnings.append("Модель не вернула оценку для: " + ", ".join(still))

        findings: List[Finding] = []
        for rid, item in results.items():
            r = by_id.get(rid)
            if not r:
                continue
            r.assessment = str(item.get("assessment") or "").strip()
            for f in item.get("findings") or []:
                fd = self._to_finding(doc, f, "requirement", "check_requirements", [rid], set(by_id))
                if fd:
                    findings.append(fd)
        self._detail(f"оценено {len(results)}/{len(reqs)} требований, находок {len(findings)}")
        return findings

    def _run_batch(self, doc, reqs, batch, rule_findings, results) -> None:
        try:
            data = self._ask(
                "check_requirements",
                prompts.check_requirements_prompt(doc, reqs, batch, rule_findings),
                schemas.CHECK_REQ_SCHEMA,
            )
        except LLMTruncated:
            if len(batch) == 1:
                self.warnings.append(f"Ответ по {batch[0].id} обрезан моделью — требование не оценено.")
                return
            half = len(batch) // 2
            self.progress(f"      ответ обрезан — делю пакет на {half} + {len(batch) - half}")
            self._run_batch(doc, reqs, batch[:half], rule_findings, results)
            self._run_batch(doc, reqs, batch[half:], rule_findings, results)
            return
        except LLMFatalError:
            raise
        except LLMError as exc:
            # сбой одного пакета не должен терять результаты остальных
            self.warnings.append(f"Пакет {batch[0].id}…{batch[-1].id} не проверен: {exc}")
            return
        batch_ids = {r.id for r in batch}
        for item in (data or {}).get("results") or []:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("req_id") or "").strip()
            if rid in batch_ids:
                results[rid] = item

    # --------------------------------------------------------------- set
    def _check_set_consistency(self, doc, reqs, known):
        data = self._ask("check_set_consistency", prompts.check_set_consistency_prompt(doc, reqs, known),
                         schemas.CHECK_SET_CONSISTENCY_SCHEMA) or {}
        ids = {r.id for r in reqs}
        findings = [f for f in (self._to_finding(doc, x, "set", "check_set_consistency", [], ids)
                                for x in data.get("findings") or []) if f]
        glossary = []
        for g in data.get("glossary") or []:
            if not isinstance(g, dict):
                continue
            variants = []
            for v in g.get("variants") or []:
                ev = self._verify_evidence(doc, v)
                if ev and all(ev.line_start != x.line_start or normalize(ev.quote) != normalize(x.quote) for x in variants):
                    variants.append(ev)
            if len(variants) >= 2:
                glossary.append(GlossaryEntry(str(g.get("concept") or "").strip(), variants,
                                              str(g.get("recommendation") or "").strip()))
        self._detail(f"находок {len(findings)}, терминов с синонимами {len(glossary)}")
        return findings, glossary

    def _check_set_completeness(self, doc, reqs, known):
        data = self._ask("check_set_completeness", prompts.check_set_completeness_prompt(doc, reqs, known),
                         schemas.CHECK_SET_COMPLETENESS_SCHEMA) or {}
        ids = {r.id for r in reqs}
        findings = [f for f in (self._to_finding(doc, x, "set", "check_set_completeness", [], ids)
                                for x in data.get("findings") or []) if f]
        trace = []
        for t in data.get("traceability") or []:
            if not isinstance(t, dict):
                continue
            status = str(t.get("status") or "partial")
            trace.append(TraceRow(
                source=str(t.get("source") or "").strip(),
                title="",
                ui=_ids_only(t.get("ui"), ids),
                api=_ids_only(t.get("api"), ids),
                data=_ids_only(t.get("data"), ids),
                acceptance=_ids_only(t.get("acceptance"), ids),
                status=status if status in ("covered", "partial", "missing") else "partial",
                note=str(t.get("note") or "").strip(),
            ))
        by_id = {r.id: r for r in reqs}
        for row in trace:
            r = by_id.get(row.source)
            row.title = (f"{r.doc_id}: " if r and r.doc_id else "") + (r.summary if r else row.source)
        trace = [row for row in trace if row.source in by_id and by_id[row.source].kind in _TRACE_KINDS]
        seen_src = set()
        trace = [row for row in trace if not (row.source in seen_src or seen_src.add(row.source))]
        missing = []
        for m in data.get("missing_scenarios") or []:
            if not isinstance(m, dict) or not m.get("scenario"):
                continue
            missing.append(MissingScenario(
                scenario=str(m.get("scenario")).strip(),
                why=str(m.get("why") or "").strip(),
                req_ids=[x for x in (m.get("related_req_ids") or []) if x in ids],
                given=str(m.get("given") or "").strip(),
                when=str(m.get("when") or "").strip(),
                then=str(m.get("then") or "").strip(),
            ))
        self._detail(f"находок {len(findings)}, строк трассировки {len(trace)}, сценариев {len(missing)}")
        return findings, trace, missing

    # ------------------------------------------------------------ review
    def _review(self, doc, reqs, findings: List[Finding]):
        data = self._ask("review", prompts.review_prompt(doc, reqs, findings), schemas.REVIEW_SCHEMA) or {}
        by_uid = {f.uid: f for f in findings}
        decisions = {}
        for d in data.get("decisions") or []:
            if isinstance(d, dict) and str(d.get("id")) in by_uid:
                decisions[str(d.get("id"))] = d
        drops = [u for u, d in decisions.items() if d.get("action") == "drop"]
        if len(drops) > 0.5 * len(findings):
            self.warnings.append(f"Ревьюер предложил удалить {len(drops)} из {len(findings)} находок — "
                                 "это подозрительно много, удаления не применены.")
            for u in drops:
                decisions[u]["action"] = "keep"

        # severity
        for u, d in decisions.items():
            sev = d.get("severity")
            if sev in SEVERITIES and sev != by_uid[u].severity:
                by_uid[u].severity = sev
                self.stats["severity_changed_review"] += 1
            if d.get("reason"):
                by_uid[u].review_note = str(d.get("reason"))[:300]

        dropped = set()
        for u, d in decisions.items():
            if d.get("action") == "drop":
                dropped.add(u)
                self.rejected.append((by_uid[u].title, "ревьюер: " + str(d.get("reason") or "")[:200]))
        self.stats["dropped_review"] += len(dropped)

        # merges (с разрешением цепочек)
        target_of = {}
        for u, d in decisions.items():
            if d.get("action") == "merge":
                t = str(d.get("merge_into") or "")
                if t in by_uid and t != u and t not in dropped:
                    target_of[u] = t

        def root(u: str) -> str:
            seen = set()
            while u in target_of and u not in seen:
                seen.add(u)
                u = target_of[u]
            return u

        kept: List[Finding] = []
        for f in findings:
            if f.uid in dropped:
                continue
            r = root(f.uid)
            if r != f.uid and r in by_uid and r not in dropped:
                continue
            kept.append(f)
        kept_by_uid = {f.uid: f for f in kept}
        for u in target_of:
            r = root(u)
            if u in dropped or r == u or r not in kept_by_uid:
                continue
            _merge_into(kept_by_uid[r], by_uid[u])
            self.stats["merged_review"] += 1

        self._detail(f"оставлено {len(kept)}, объединено {self.stats['merged_review']}, отброшено {len(dropped)}")
        return (
            kept,
            [p for p in data.get("priorities") or [] if isinstance(p, dict)],
            str(data.get("summary") or "").strip(),
            str(data.get("development_reason") or "").strip(),
            str(data.get("testing_reason") or "").strip(),
        )

    # ---------------------------------------------------------- findings
    def _verify_evidence(self, doc: Document, item) -> Optional[Evidence]:
        if not isinstance(item, dict):
            return None
        quote = str(item.get("quote") or "")
        try:
            hint = int(item.get("line")) if item.get("line") is not None else None
        except (TypeError, ValueError):
            hint = None
        self.stats["quotes_total"] += 1
        ev = doc.locate(quote, hint)
        if ev is None:
            self.stats["quotes_rejected"] += 1
            return None
        self.stats["quotes_exact" if ev.match == "exact" else "quotes_fuzzy"] += 1
        return ev

    def _to_finding(self, doc, item, scope, stage, req_ids, known_ids) -> Optional[Finding]:
        if not isinstance(item, dict):
            return None
        title = str(item.get("title") or "").strip()
        evidence: List[Evidence] = []
        for e in item.get("evidence") or []:
            ev = self._verify_evidence(doc, e)
            if ev and not any(x.line_start == ev.line_start and normalize(x.quote) == normalize(ev.quote) for x in evidence):
                evidence.append(ev)
        if not evidence:
            self.stats["findings_unverified"] += 1
            self.rejected.append((title or "(без названия)", "цитата не найдена в документе"))
            return None
        crit = str(item.get("criterion") or "")
        allowed = INDIVIDUAL_IDS if scope == "requirement" else SET_IDS
        if crit not in allowed:
            if scope == "requirement" and crit in _TO_REQ:
                crit = _TO_REQ[crit]
            elif scope == "set" and crit in _TO_SET:
                crit = _TO_SET[crit]
            else:
                crit = "complete" if scope == "requirement" else "set_complete"
        sev = str(item.get("severity") or "major")
        if sev not in SEVERITIES:
            sev = "major"
        q = item.get("question") or {}
        question = None
        if isinstance(q, dict) and str(q.get("text") or "").strip():
            to = str(q.get("to") or "analyst")
            question = Question(to if to in ADDRESSEES else "analyst", str(q.get("text")).strip())
        related = [str(x).strip() for x in (item.get("related_req_ids") or []) if str(x).strip() in known_ids]
        ids = list(dict.fromkeys(list(req_ids) + related))
        return Finding(
            uid=self._next_uid("L"),
            scope=scope,
            criterion=crit,
            severity=sev,
            title=title or ALL[crit].name,
            evidence=evidence,
            explanation=str(item.get("explanation") or "").strip(),
            recommendation=str(item.get("recommendation") or "").strip(),
            question=question,
            req_ids=ids,
            test_idea=str(item.get("test_idea") or "").strip(),
            source="llm",
            stage=stage,
        )

    def _attach(self, doc: Document, reqs: List[Requirement], findings: List[Finding]) -> List[Finding]:
        """Привязка находок к требованиям по строкам цитат."""
        for f in findings:
            mapped = []
            for e in f.evidence:
                r = find_requirement(reqs, e.line_start) or find_requirement(reqs, e.line_end)
                if r and r.id not in mapped:
                    mapped.append(r.id)
            if f.source == "rule":
                if f.criterion in INDIVIDUAL_IDS:
                    if mapped:
                        f.scope = "requirement"
                        f.req_ids = mapped
                    else:
                        f.scope = "set"
                        f.criterion = _TO_SET.get(f.criterion, "set_complete")
                else:
                    f.scope = "set"
                    f.req_ids = mapped
            elif f.scope == "set":
                f.req_ids = list(dict.fromkeys(f.req_ids + mapped))
            else:
                f.req_ids = list(dict.fromkeys(f.req_ids + [m for m in mapped if m not in f.req_ids]))
        return findings

    def _auto_merge(self, findings: List[Finding]) -> List[Finding]:
        """Детерминированная склейка очевидных дублей (правило ↔ LLM, LLM ↔ LLM)."""
        out: List[Finding] = []
        for f in sorted(findings, key=lambda x: 0 if x.source == "rule" else 1):
            target = None
            best = -1.0
            f_lines = {e.line_start for e in f.evidence}
            for g in out:
                if _FAMILY.get(f.criterion) != _FAMILY.get(g.criterion):
                    continue
                g_lines = {e.line_start for e in g.evidence}
                if not (f_lines & g_lines):
                    continue
                sim = _jaccard(f.title + " " + f.explanation[:200], g.title + " " + g.explanation[:200])
                if f.source != g.source:
                    # находка LLM целиком внутри строк правила (или наоборот) того же семейства — это та же проблема
                    ok = sim >= 0.2 or ((f_lines <= g_lines or g_lines <= f_lines) and sim >= 0.08)
                else:
                    ok = f.primary_req == g.primary_req and _jaccard(f.title, g.title) >= 0.6
                if ok and sim > best:
                    best, target = sim, g
            if target:
                _merge_into(target, f)
                self.stats["merged_auto"] += 1
            else:
                out.append(f)
        return out

    def _merge_subsumed_contradictions(self, findings: List[Finding]) -> List[Finding]:
        """Противоречие, все цитаты и требования которого входят в другое противоречие, — его дубль
        (одну и ту же коллизию модель могла описать с двух сторон)."""
        cons = [f for f in findings if _FAMILY.get(f.criterion) == "cons"]
        absorbed = set()
        for f in sorted(cons, key=lambda x: len({e.line_start for e in x.evidence})):
            f_lines = {e.line_start for e in f.evidence}
            for g in cons:
                if g is f or g.uid in absorbed or f.uid in absorbed:
                    continue
                g_lines = {e.line_start for e in g.evidence}
                if f_lines <= g_lines and len(g_lines) > len(f_lines) and set(f.req_ids) <= set(g.req_ids):
                    _merge_into(g, f)
                    absorbed.add(f.uid)
                    self.stats["merged_auto"] += 1
                    break
        return [f for f in findings if f.uid not in absorbed]

    def _number(self, findings: List[Finding], reqs: List[Requirement]) -> List[Finding]:
        order = {r.id: i for i, r in enumerate(reqs)}
        set_order = {c: i for i, c in enumerate(SET_IDS)}

        def key(f: Finding):
            if f.scope == "requirement":
                return (0, order.get(f.primary_req, 10 ** 4), f.rank, f.first_line)
            return (1, set_order.get(f.criterion, 99), f.rank, f.first_line)

        findings.sort(key=key)
        width = 3
        for i, f in enumerate(findings, start=1):
            f.id = f"D-{i:0{width}d}"
        return findings


def _merge_into(target: Finding, other: Finding) -> None:
    if SEVERITY_RANK.get(other.severity, 9) < SEVERITY_RANK.get(target.severity, 9):
        target.severity = other.severity
    if target.source == "rule" and other.source == "llm":
        # заголовок и вопрос правила точны; конкретику LLM берём, только если она про то же самое
        if _jaccard(target.title + " " + target.explanation, other.title + " " + other.explanation) >= 0.15:
            if len(other.explanation) > len(target.explanation):
                target.explanation = other.explanation
            if other.recommendation and len(other.recommendation) > len(target.recommendation):
                target.recommendation = other.recommendation
        target.source = "rule+llm"
    elif other.rule_id and not target.rule_id:
        # находку правила поглотила находка LLM — сохраняем след правила
        target.rule_id = other.rule_id
        target.source = "rule+llm"
    for e in other.evidence:
        if not any(x.line_start == e.line_start for x in target.evidence):
            target.evidence.append(e)
    for r in other.req_ids:
        if r not in target.req_ids:
            target.req_ids.append(r)
    if not target.test_idea and other.test_idea:
        target.test_idea = other.test_idea
    target.merged_from.append(other.uid)
    target.merged_from.extend(other.merged_from)


def _stems(text: str) -> set:
    # грубый стемминг: первые 5 символов слова — достаточно для сравнения формулировок на русском
    return {t[:5] for t in re.findall(r"[0-9a-zа-я]+", normalize(text)) if len(t) > 3}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _stems(a), _stems(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _ids_only(value, known) -> str:
    found = re.findall(r"REQ-\d+", str(value or ""))
    ok = [x for x in dict.fromkeys(found) if x in known]
    return ", ".join(ok) if ok else "—"


def run_audit(path: Path, llm: Optional[LLMClient], batch_size: int = 10, review_pass: bool = True,
              progress: Optional[Callable[[str], None]] = None) -> AuditResult:
    return Auditor(llm, batch_size=batch_size, review_pass=review_pass, progress=progress).run(Path(path))
