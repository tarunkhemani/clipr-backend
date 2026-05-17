import os


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_caption_chunks(word_timestamps: list, words_per_chunk: int = 4) -> list:
    """
    Groups word timestamps into caption chunks.
    Each chunk = 3-4 words, displayed together.
    This is the CapCut-style caption look.
    Returns list of { start, end, text } dicts.
    """
    if not word_timestamps:
        return []

    chunks = []
    i = 0

    while i < len(word_timestamps):
        chunk_words = word_timestamps[i:i + words_per_chunk]

        start = chunk_words[0]["start"]
        end = chunk_words[-1]["end"]
        text = " ".join(w["word"] for w in chunk_words)

        # Make sure minimum display time is 0.3s
        if end - start < 0.3:
            end = start + 0.3

        chunks.append({
            "start": start,
            "end": end,
            "text": text.strip(),
        })

        i += words_per_chunk

    return chunks


def write_srt_file(chunks: list, output_path: str) -> str:
    """
    Writes caption chunks to a proper SRT file.
    Returns the path to the SRT file.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_time(chunk['start'])} --> {seconds_to_srt_time(chunk['end'])}")
        lines.append(chunk["text"])
        lines.append("")  # Blank line between entries

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def build_srt_from_words(word_timestamps: list, output_path: str, words_per_chunk: int = 4) -> str:
    """
    Full pipeline: word timestamps → grouped chunks → SRT file.
    Main function to call from outside.
    """
    chunks = build_caption_chunks(word_timestamps, words_per_chunk)
    return write_srt_file(chunks, output_path)