import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import requests

from src.schema import SpecItem, EvidenceChunk, ExtractionResponse

logger = logging.getLogger(__name__)


class LLMExtractor:
    """
    Extracts structured vehicle specifications from retrieved evidence passages.
    Supports both optional LLM APIs (OpenAI/compatible) and a robust offline
    rule-based extraction fallback for deterministic automotive parsing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    @property
    def has_llm_credentials(self) -> bool:
        """Returns True if a non-empty API key is available."""
        return bool(self.api_key and len(self.api_key) > 5)

    def extract(
        self,
        query: str,
        evidence_chunks: List[EvidenceChunk],
        use_llm: bool = True,
        document_name: Optional[str] = None
    ) -> ExtractionResponse:
        """
        Main extraction entry point. Attempts LLM extraction if enabled and keys exist;
        otherwise falls back to rule-based pattern extraction.
        """
        if not evidence_chunks:
            return ExtractionResponse(
                query=query,
                status="not_found",
                specifications=[],
                evidence_chunks=[],
                extraction_method="not_found",
                message="No relevant passages were found in the document for this query.",
                document_name=document_name
            )

        # Tier 1: Try LLM if enabled and credentials present
        if use_llm and self.has_llm_credentials:
            try:
                response = self._extract_with_llm(query, evidence_chunks, document_name)
                if response.specifications:
                    return response
            except Exception as e:
                logger.warning(f"LLM extraction encountered an error ({e}), falling back to rule-based: {e}")

        # Tier 2: Deterministic Rule-Based Fallback
        return self._extract_with_rules(query, evidence_chunks, document_name)

    def _extract_with_llm(
        self,
        query: str,
        evidence_chunks: List[EvidenceChunk],
        document_name: Optional[str] = None
    ) -> ExtractionResponse:
        """Invokes LLM API with strict JSON schema instructions and grounded validation."""
        context_blocks = []
        for c in evidence_chunks:
            context_blocks.append(
                f"[Page {c.page_number} | Chunk {c.chunk_id}]\n{c.text}\n"
            )
        context_str = "\n---\n".join(context_blocks)

        system_prompt = (
            "You are an expert automotive technical data extractor. "
            "Your task is to extract exact vehicle specifications from the provided service manual excerpts.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Ground all extractions STRICTLY in the provided text excerpts.\n"
            "2. Never hallucinate, invent values, or assume standard numbers.\n"
            "3. If multiple models, engines, or procedures have different specs, output each as a separate item.\n"
            "4. Extract ONLY specifications for the component explicitly asked for in the user's query. "
            "DO NOT extract neighboring components or adjacent table rows (e.g., if asked for 'Brake disc shield bolts', "
            "do NOT include 'Brake hose bracket bolt').\n"
            "5. The 'page_number' MUST be the 1-indexed PDF page number stated in the excerpt header '[Page X | ...]'.\n"
            "6. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "specifications": [\n'
            '    {\n'
            '      "component": "Component name (e.g., Brake Caliper Bolt)",\n'
            '      "spec_type": "Spec type (e.g., Torque, Fluid Capacity, Clearance, Part Number)",\n'
            '      "value": "Exact number or range (e.g., 35, 17 - 150)",\n'
            '      "unit": "Unit (e.g., Nm, lb-ft, lb-in, L, qt, mm, psi)",\n'
            '      "page_number": 24,\n'
            '      "source_context": "Exact sentence or row from excerpt",\n'
            '      "notes": "Variant/context notes (e.g., RWD, 4WD, 5.0L engine)"\n'
            "    }\n"
            "  ],\n"
            '  "status": "success" or "not_found",\n'
            '  "message": "Brief summary of extraction"\n'
            "}\n"
            "If no specification is present in the text for the query, return specifications: [] and status: 'not_found'."
        )

        user_content = f"QUERY: {query}\n\nEXCERPTS FROM SERVICE MANUAL:\n{context_str}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        raw_reply = data["choices"][0]["message"]["content"]

        parsed = json.loads(raw_reply)
        specs: List[SpecItem] = []
        valid_pages = {c.page_number for c in evidence_chunks}
        all_evidence_text = " ".join(c.text for c in evidence_chunks)

        for item in parsed.get("specifications", []):
            comp = item.get("component", "").strip()
            val = str(item.get("value", "")).strip()
            unit = item.get("unit", "").strip()
            page_num = item.get("page_number")

            if not comp or not val:
                continue

            # Grounding validation: Value must exist in retrieved evidence text
            if val not in all_evidence_text and val.replace(".", ",") not in all_evidence_text:
                logger.warning(f"Discarding ungrounded LLM hallucination: {comp} value={val}")
                continue

            # Page provenance validation
            if page_num is not None and page_num not in valid_pages:
                matched_page = None
                for c in evidence_chunks:
                    if val in c.text:
                        matched_page = c.page_number
                        break
                page_num = matched_page or (evidence_chunks[0].page_number if evidence_chunks else None)

            # Component relevance verification: reject adjacent rows
            comp_score = self.compute_component_match_score(query, comp)
            if comp_score < 0.45:
                logger.warning(f"Discarding irrelevant/adjacent component from LLM: {comp} (score={comp_score})")
                continue

            specs.append(SpecItem(
                component=comp,
                spec_type=item.get("spec_type", "Specification"),
                value=val,
                unit=unit,
                page_number=page_num,
                source_context=item.get("source_context", ""),
                confidence=round(min(0.95, 0.70 + 0.25 * comp_score), 2),
                notes=item.get("notes")
            ))

        status = "success" if specs else "not_found"
        msg = parsed.get("message", "Extracted using LLM.") if specs else "No specification found in the retrieved manual passages."

        return ExtractionResponse(
            query=query,
            status=status,
            specifications=specs,
            evidence_chunks=evidence_chunks,
            extraction_method="llm",
            message=msg,
            document_name=document_name
        )

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Normalizes plurals and inflections for robust technical matching."""
        t = token.lower().strip()
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        if t.endswith("es") and len(t) > 4 and t[-3] in ("s", "x", "z", "c", "h"):
            return t[:-2]
        if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            return t[:-1]
        return t

    @classmethod
    def compute_component_match_score(cls, query: str, comp_name: str) -> float:
        """
        Computes precision match score between a user query and a candidate component name.
        Detects antonym collisions (e.g., upper vs lower), normalizes stems (bolts -> bolt),
        and combines token containment with Jaccard overlap to prevent adjacent-row false positives.
        """
        q_lower = query.lower()
        c_lower = comp_name.lower()

        # Stop words that describe questions rather than the component itself
        stop = {
            'what', 'is', 'the', 'for', 'of', 'in', 'to', 'at', 'a', 'an',
            'torque', 'spec', 'specs', 'specification', 'specifications',
            'value', 'values', 'tighten', 'tightening', 'capacity', 'capacities',
            'fluid', 'part', 'number', 'base', 'item', 'approximate', 'refill',
            'give', 'find', 'me', 'how', 'much', 'please'
        }
        raw_q_tokens = [w for w in re.findall(r'\b[a-z0-9]+\b', q_lower) if w not in stop]
        raw_c_tokens = [w for w in re.findall(r'\b[a-z0-9]+\b', c_lower)]

        if not raw_q_tokens or not raw_c_tokens:
            return 0.0

        # Antonym collision detection: immediately disqualify conflicting components
        antonyms = [
            ('upper', 'lower'),
            ('front', 'rear'),
            ('inner', 'outer'),
            ('left', 'right'),
            ('forward', 'rearward'),
            ('inlet', 'outlet')
        ]
        for a1, a2 in antonyms:
            if (a1 in raw_q_tokens and a2 in raw_c_tokens) or (a2 in raw_q_tokens and a1 in raw_c_tokens):
                return 0.0

        # Stem normalization (e.g. bolts -> bolt, nuts -> nut)
        q_tokens = [cls._normalize_token(w) for w in raw_q_tokens]
        c_tokens = [cls._normalize_token(w) for w in raw_c_tokens]

        q_set = set(q_tokens)
        c_set = set(c_tokens)

        # Exact phrase match
        target_phrase = " ".join(q_tokens)
        c_phrase = " ".join(c_tokens)
        if target_phrase == c_phrase:
            return 1.0
        if target_phrase in c_phrase:
            len_ratio = len(q_tokens) / len(c_tokens)
            return 0.90 + 0.10 * len_ratio

        overlap = q_set.intersection(c_set)
        if not overlap:
            return 0.0

        containment = len(overlap) / len(q_set)
        union = q_set.union(c_set)
        jaccard = len(overlap) / len(union) if union else 0.0

        # Hybrid balance: rewards containing the query terms while penalizing extra unrelated words
        score = 0.65 * containment + 0.35 * jaccard
        return round(score, 3)

    def _extract_with_rules(
        self,
        query: str,
        evidence_chunks: List[EvidenceChunk],
        document_name: Optional[str] = None
    ) -> ExtractionResponse:
        """
        Deterministic, offline extraction using automotive regular expressions,
        precise component scoring, and multi-column tabular parsing.
        """
        scored_specs: List[tuple[SpecItem, float]] = []
        query_lower = query.lower()

        is_part_query = any(w in query_lower for w in ["part", "number", "kit"])
        is_capacity_query = any(w in query_lower for w in ["capacity", "fluid", "fill", "oil", "coolant", "lubricant", "liters", "quarts", "pt"])

        for chunk in evidence_chunks:
            text = chunk.text
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            # -------------------------------------------------------------
            # Strategy A: Multi-Column Torque Specification Tables (e.g. Page 24)
            # -------------------------------------------------------------
            header_idx = -1
            for idx, l in enumerate(lines):
                if 'nm' in l.lower() and ('lb-ft' in l.lower() or 'lb-in' in l.lower()):
                    header_idx = idx
                    break

            if header_idx != -1 and not is_part_query and not is_capacity_query:
                i = header_idx + 1
                while i < len(lines):
                    line = lines[i]
                    if any(ignore in line.lower() for ignore in ['section', 'workshop manual', 'procedure revision', 'description', 'page']):
                        i += 1
                        continue

                    # Check if line looks like a component description
                    if re.match(r'^[A-Za-z][A-Za-z0-9\s\-\/\(\)]+$', line) and not any(u in line for u in ['Nm', 'lb-ft', 'lb-in']):
                        comp = line
                        nm_override = None

                        # Check if line ended with trailing number from PDF stream (e.g. 'bracket bolt 12')
                        trailing_num_match = re.match(r'^(.*?)\s+(\d+(?:\.\d+)?)$', comp)
                        if trailing_num_match:
                            potential_comp = trailing_num_match.group(1).strip()
                            if len(potential_comp) > 4:
                                comp = potential_comp
                                nm_override = trailing_num_match.group(2)

                        j = i + 1
                        tokens = []
                        if nm_override:
                            tokens.append(nm_override)

                        while j < len(lines):
                            next_l = lines[j]
                            if any(ignore in next_l.lower() for ignore in ['section', 'workshop manual', 'procedure revision', 'page']):
                                j += 1
                                continue
                            if re.match(r'^[A-Za-z][A-Za-z0-9\s\-\/\(\)]+$', next_l) and not any(u in next_l for u in ['Nm', 'lb-ft', 'lb-in']):
                                words = [w for w in next_l.split() if re.match(r'^[A-Za-z]', w)]
                                if len(words) >= 1 and not re.match(r'^\d', next_l):
                                    break

                            parts = re.findall(r'(\d+(?:\.\d+)?|[-–—\ufffd])', next_l)
                            if parts:
                                tokens.extend(parts)
                                j += 1
                                if len(tokens) >= 3:
                                    break
                            else:
                                break

                        if tokens:
                            # Map columns: 0=Nm, 1=lb-ft, 2=lb-in
                            nm_val = tokens[0] if len(tokens) > 0 and tokens[0] not in ('-', '–', '—', '\ufffd') else None
                            lbft_val = tokens[1] if len(tokens) > 1 and tokens[1] not in ('-', '–', '—', '\ufffd') else None
                            lbin_val = tokens[2] if len(tokens) > 2 and tokens[2] not in ('-', '–', '—', '\ufffd') else None

                            primary_val = nm_val or lbft_val or lbin_val
                            primary_unit = "Nm" if nm_val else ("lb-ft" if lbft_val else "lb-in")

                            notes_parts = []
                            if lbft_val and primary_unit != "lb-ft":
                                notes_parts.append(f"{lbft_val} lb-ft")
                            if lbin_val and primary_unit != "lb-in":
                                notes_parts.append(f"{lbin_val} lb-in")

                            notes = f"Alternate units: {', '.join(notes_parts)}" if notes_parts else None
                            match_score = self.compute_component_match_score(query, comp)

                            if primary_val:
                                item = SpecItem(
                                    component=comp,
                                    spec_type="Torque",
                                    value=primary_val,
                                    unit=primary_unit,
                                    page_number=chunk.page_number,
                                    source_context=f"{comp} | Nm: {nm_val or '-'} | lb-ft: {lbft_val or '-'} | lb-in: {lbin_val or '-'}",
                                    confidence=round(0.90 + 0.08 * match_score, 2),
                                    notes=notes
                                )
                                scored_specs.append((item, match_score))

                            i = j
                            continue
                    i += 1

            # -------------------------------------------------------------
            # Strategy B: Part Number Tables & Inline Kit Mentions (e.g. Page 132 & Page 12)
            # -------------------------------------------------------------
            part_header_idx = -1
            for idx, l in enumerate(lines):
                if 'part number' in l.lower() or 'part #' in l.lower():
                    part_header_idx = idx
                    break

            if part_header_idx != -1:
                p_i = part_header_idx + 1
                while p_i < len(lines) - 2:
                    l1, l2, l3 = lines[p_i], lines[p_i+1], lines[p_i+2]
                    # Pattern: <Item_Number> \n <Part_Number> \n <Description>
                    if re.match(r'^\d+$', l1) and re.match(r'^[A-Z0-9\-]{4,15}$', l2, re.I) and re.match(r'^[A-Za-z]', l3):
                        p_score = self.compute_component_match_score(query, l3)
                        scored_specs.append((SpecItem(
                            component=l3,
                            spec_type="Part Number",
                            value=l2,
                            unit="Part #",
                            page_number=chunk.page_number,
                            source_context=f"Item {l1}: {l3} (Part Number: {l2})",
                            confidence=0.95,
                            notes=f"Item number: {l1}"
                        ), p_score))
                        p_i += 3
                        continue
                    p_i += 1

            # Inline kits/parts: e.g. 'cam bolt kit 3C333'
            inline_kits = re.findall(r'([A-Za-z0-9\s\-\/\(\)]{3,40}?kit)\s+([A-Za-z0-9\-]{4,12})', text, re.I)
            for c_kit, p_num in inline_kits:
                c_clean = c_kit.strip().strip(":-•* ")
                k_score = self.compute_component_match_score(query, c_clean)
                scored_specs.append((SpecItem(
                    component=c_clean,
                    spec_type="Part Number",
                    value=p_num.strip(),
                    unit="Part #",
                    page_number=chunk.page_number,
                    source_context=f"{c_clean} {p_num}",
                    confidence=0.90,
                    notes="Inline service part reference"
                ), k_score))

            # -------------------------------------------------------------
            # Strategy C: Fluid Capacities & Procedural Refills (e.g. Page 258 & Page 308)
            # -------------------------------------------------------------
            # Procedural: 'fill the axle with 2.8L (6.0 pt) of axle lubricant'
            proc_fills = re.findall(
                r'(?:fill|capacity\s+of|capacity:?)\s*(?:the)?\s*([A-Za-z0-9\s\-\/]{3,35}?)\s*(?:with|is|:)?\s*(\d+(?:\.\d+)?)\s*(L|liters?|quarts?|qt|gal|pints?|pt)\s*(?:\(([\d\.]+\s*(?:pt|qt|L|gal))\))?',
                text,
                re.I
            )
            for sys_name, val, unit, alt in proc_fills:
                clean_sys = sys_name.strip()
                if "axle" in clean_sys.lower():
                    clean_sys = "Rear Axle Lubricant"
                cap_score = self.compute_component_match_score(query, clean_sys)
                notes_alt = f"Alternate units: {alt}" if alt else None
                scored_specs.append((SpecItem(
                    component=clean_sys,
                    spec_type="Fluid Capacity",
                    value=val.strip(),
                    unit=unit.strip(),
                    page_number=chunk.page_number,
                    source_context=f"Fill {clean_sys} with {val} {unit}" + (f" ({alt})" if alt else ""),
                    confidence=0.92,
                    notes=notes_alt
                ), cap_score))

            # Table specifications: '2.84L (6.0 pt) ... Synthetic Rear Axle Lubricant'
            table_caps = re.findall(
                r'([A-Za-z0-9\s\-\/\(\)]{5,50}?(?:Lubricant|Fluid|Oil|Coolant))\s*.*?(?<![A-Za-z0-9\-])(\d+(?:\.\d+)?)\s*(L|liters?|quarts?|qt|gal|pints?|pt)\b(?:\s*\(([\d\.]+\s*(?:pt|qt|L|gal))\))?',
                text,
                re.I | re.DOTALL
            )
            for mat_name, val, unit, alt in table_caps:
                mat_clean = " ".join(mat_name.split())
                # Strip leading table header remnants such as "pacity", "capacity", "material", "fill"
                mat_clean = re.sub(r'^(?:fill\s+|capacity\s+|pacity\s+|material\s+|item\s+|specification\s+|spec\s+)+', '', mat_clean, flags=re.I).strip()
                if "rear axle" in mat_clean.lower() or "synthetic rear axle" in mat_clean.lower():
                    mat_clean = "Rear Axle Lubricant"
                cap_score = self.compute_component_match_score(query, mat_clean)
                notes_alt = f"Alternate units: {alt}" if alt else None
                scored_specs.append((SpecItem(
                    component=mat_clean,
                    spec_type="Fluid Capacity",
                    value=val.strip(),
                    unit=unit.strip(),
                    page_number=chunk.page_number,
                    source_context=f"{mat_clean}: {val} {unit}" + (f" ({alt})" if alt else ""),
                    confidence=0.92,
                    notes=notes_alt
                ), cap_score))

            # -------------------------------------------------------------
            # Strategy D: Procedural Tightening Lines (e.g. Page 49, 14)
            # -------------------------------------------------------------
            for line_str in lines:
                inline_torque = re.findall(
                    r'([A-Za-z0-9\s\-\(\)\/]{3,40}?)\s*(?:tighten to|tighten the nut to|torque(?:\s+to)?|at)\s*(\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?)\s*(Nm|N·m)\s*(?:\(([\d\.]+\s*(?:lb-ft|lb-in))\))?',
                    line_str,
                    re.IGNORECASE
                )
                for comp_raw, val, unit, alt in inline_torque:
                    c_clean = comp_raw.strip().strip(":-•* ")
                    if not c_clean or len(c_clean) < 3 or c_clean.lower() in ("to install", "install", "and"):
                        # Extract antecedent or fallback
                        c_clean = "Specified Fastener"
                    m_score = self.compute_component_match_score(query, c_clean)
                    notes_alt = f"Alternate units: {alt}" if alt else "Extracted from procedure instruction"
                    scored_specs.append((SpecItem(
                        component=c_clean,
                        spec_type="Torque",
                        value=val.strip(),
                        unit=unit.strip(),
                        page_number=chunk.page_number,
                        source_context=line_str,
                        confidence=0.88,
                        notes=notes_alt
                    ), m_score))

        # -------------------------------------------------------------
        # Precision Filtering: Discard Unrelated Rows
        # -------------------------------------------------------------
        if not scored_specs:
            return ExtractionResponse(
                query=query,
                status="not_found",
                specifications=[],
                evidence_chunks=evidence_chunks,
                extraction_method="rule_based_fallback",
                message="No matching specification found in the retrieved manual passages.",
                document_name=document_name
            )

        max_score = max(score for _, score in scored_specs)

        # A specification is accepted only if it demonstrates strong relevance
        threshold = 0.70 if max_score >= 0.70 else 0.50

        filtered_specs: List[SpecItem] = []
        seen = set()

        for s, score in scored_specs:
            # Must meet minimum threshold and be in the top-scoring tier
            if score >= threshold and score >= (max_score - 0.05):
                key = (s.component.lower(), s.spec_type.lower(), s.value, s.unit.lower(), s.page_number)
                if key not in seen:
                    seen.add(key)
                    filtered_specs.append(s)

        if filtered_specs and max_score >= 0.65:
            return ExtractionResponse(
                query=query,
                status="success",
                specifications=filtered_specs,
                evidence_chunks=evidence_chunks,
                extraction_method="rule_based_fallback",
                message=f"Extracted {len(filtered_specs)} specification(s) via deterministic rule-based analysis.",
                document_name=document_name
            )
        else:
            return ExtractionResponse(
                query=query,
                status="not_found",
                specifications=[],
                evidence_chunks=evidence_chunks,
                extraction_method="rule_based_fallback",
                message="No matching specification found in the retrieved manual passages.",
                document_name=document_name
            )
