from typing import List, Dict, Any, Optional


class DocumentChunker:
    """
    Splits document pages into clean, line-aligned, overlapping text chunks.
    Ensures that text is never cut mid-word or mid-line and preserves
    technical table rows intact.
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str = "doc"
    ) -> List[Dict[str, Any]]:
        """
        Splits a list of cleaned page dicts into line-aligned overlapping text chunks.
        Never cuts across words or lines.
        """
        all_chunks: List[Dict[str, Any]] = []

        for p in pages:
            page_num = p.get("page_number", 1)
            raw_text = p.get("text", "").strip()

            if not raw_text:
                continue

            # If page text fits cleanly within chunk_size, preserve it completely as 1 chunk
            if len(raw_text) <= self.chunk_size:
                all_chunks.append({
                    "chunk_id": f"{doc_id}_p{page_num}_c0",
                    "page_number": page_num,
                    "text": raw_text,
                    "start_idx": 0,
                    "end_idx": len(raw_text),
                    "chunk_index": 0
                })
                continue

            # Split into individual lines to guarantee no half-words or chopped table rows
            lines = [l for l in raw_text.split("\n")]
            chunk_idx = 0
            i = 0
            start_line_idx = 0
            current_lines = []
            current_len = 0

            # Detect if page starts with a table header or section title to propagate
            table_header = ""
            for h_candidate in lines[:6]:
                if any(kw in h_candidate.lower() for kw in ["torque specifications", "nm lb-ft", "part number", "fill capacity", "description"]):
                    table_header += h_candidate + "\n"

            while i < len(lines):
                line = lines[i]
                line_len = len(line) + 1  # newline

                # If adding line exceeds chunk_size and we already accumulated lines
                if current_len + line_len > self.chunk_size and current_lines:
                    chunk_text = "\n".join(current_lines).strip()
                    if chunk_text:
                        all_chunks.append({
                            "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                            "page_number": page_num,
                            "text": chunk_text,
                            "start_idx": start_line_idx,
                            "end_idx": i,
                            "chunk_index": chunk_idx
                        })
                        chunk_idx += 1

                    # Backtrack lines for overlap without cutting words
                    backtrack_len = 0
                    k = i - 1
                    while k >= start_line_idx and backtrack_len + len(lines[k]) + 1 <= self.chunk_overlap:
                        backtrack_len += len(lines[k]) + 1
                        k -= 1

                    new_start = max(k + 1, start_line_idx + 1)
                    i = new_start
                    start_line_idx = new_start
                    current_lines = []
                    current_len = 0

                    # Propagate table header context if chunk split a table
                    if table_header and table_header.strip() not in lines[i]:
                        current_lines.append(table_header.strip())
                        current_len += len(table_header)
                    continue

                current_lines.append(line)
                current_len += line_len
                i += 1

            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    all_chunks.append({
                        "chunk_id": f"{doc_id}_p{page_num}_c{chunk_idx}",
                        "page_number": page_num,
                        "text": chunk_text,
                        "start_idx": start_line_idx,
                        "end_idx": len(lines),
                        "chunk_index": chunk_idx
                    })

        return all_chunks
