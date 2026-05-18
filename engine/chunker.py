"""
Smart text chunking with overlap — preserves context across chunk boundaries.
Uses tiktoken for accurate token counting.
"""

import tiktoken

from .document_loader import Document


class SemanticChunker:
    """
    Splits documents into overlapping chunks sized by token count.
    Uses paragraph/sentence boundaries when possible to avoid mid-sentence splits.
    """

    def __init__(
        self,
        max_tokens: int = 256,
        overlap_tokens: int = 64,
        tokenizer_name: str = "cl100k_base",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def chunk_document(self, doc: Document) -> list[Document]:
        text = doc.text.strip()
        if not text:
            return []

        # If the whole document fits in one chunk, return as-is
        if self.count_tokens(text) <= self.max_tokens:
            return [doc]

        # Split by paragraphs first, then by sentences if needed
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        segments: list[str] = []
        for para in paragraphs:
            if self.count_tokens(para) <= self.max_tokens:
                segments.append(para)
            else:
                # Split long paragraphs by sentences
                sentences = self._split_sentences(para)
                segments.extend(sentences)

        return self._merge_segments(segments, doc.metadata)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple heuristics."""
        import re

        parts = re.split(r"(?<=[.!?])\s+", text)
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # If a single sentence is still too long, force-split by tokens
            if self.count_tokens(part) > self.max_tokens:
                tokens = self.enc.encode(part)
                for i in range(0, len(tokens), self.max_tokens):
                    chunk_tokens = tokens[i : i + self.max_tokens]
                    result.append(self.enc.decode(chunk_tokens))
            else:
                result.append(part)
        return result

    def _merge_segments(
        self, segments: list[str], base_metadata: dict
    ) -> list[Document]:
        """Merge small segments into chunks up to max_tokens with overlap."""
        chunks: list[Document] = []
        current_segments: list[str] = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = self.count_tokens(seg)

            if current_tokens + seg_tokens > self.max_tokens and current_segments:
                # Emit current chunk
                chunk_text = "\n".join(current_segments)
                meta = {**base_metadata, "chunk_index": len(chunks)}
                chunks.append(Document(text=chunk_text, metadata=meta))

                # Overlap: keep trailing segments that fit within overlap budget
                overlap_segs: list[str] = []
                overlap_count = 0
                for s in reversed(current_segments):
                    s_tok = self.count_tokens(s)
                    if overlap_count + s_tok > self.overlap_tokens:
                        break
                    overlap_segs.insert(0, s)
                    overlap_count += s_tok

                current_segments = overlap_segs
                current_tokens = overlap_count

            current_segments.append(seg)
            current_tokens += seg_tokens

        # Final chunk
        if current_segments:
            chunk_text = "\n".join(current_segments)
            meta = {**base_metadata, "chunk_index": len(chunks)}
            chunks.append(Document(text=chunk_text, metadata=meta))

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        result = []
        for doc in docs:
            result.extend(self.chunk_document(doc))
        return result
