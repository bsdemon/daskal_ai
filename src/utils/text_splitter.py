from typing import List, Dict, Any, Optional
from src.core.config import dynamic_settings as settings


class TextSplitter:
    """Split text into chunks for embedding and retrieval."""

    def __init__(
        self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, separator: str = "\n"
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split a text into chunks of specified size with overlap."""
        # Handle empty 
        if not text:
            return []

        # Split by separator
        segments = text.split(self.separator)
        chunks = []
        current_chunk: List[str] = []
        current_size = 0

        for segment in segments:
            # Skip empty segments
            if not segment.strip():
                continue

            # Calculate segment size (approximating token count)
            segment_size = len(segment.split())

            # If adding this segment would exceed chunk size, finalize current chunk
            if current_size + segment_size > self.chunk_size and current_size > 0:
                # Join current chunk and add to chunks list
                chunks.append(self.separator.join(current_chunk))

                # Create overlap by keeping some segments from the end
                overlap_segments: List[str] = []
                overlap_size = 0

                # Start from the end and add segments until we reach chunk_overlap
                for seg in reversed(current_chunk):
                    seg_size = len(seg.split())
                    if overlap_size + seg_size <= self.chunk_overlap:
                        overlap_segments.insert(0, seg)
                        overlap_size += seg_size
                    else:
                        break

                # Start new chunk with overlap segments
                current_chunk = overlap_segments
                current_size = overlap_size

            # Add segment to current chunk
            current_chunk.append(segment)
            current_size += segment_size

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))

        return chunks

    def split_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        metadata_key: str = "metadata",
    ) -> List[Dict[str, Any]]:
        """Split documents into chunks, preserving metadata."""
        chunked_documents = []

        for doc in documents:
            # Validate document structure
            if text_key not in doc:
                continue

            text = doc[text_key]
            metadata = doc.get(metadata_key, {})

            # Split text into chunks
            chunks = self.split_text(text)

            # Create new document for each chunk
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)

                chunked_documents.append(
                    {text_key: chunk, metadata_key: chunk_metadata}
                )

        return chunked_documents
