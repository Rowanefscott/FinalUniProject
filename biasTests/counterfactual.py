# Swaps gendered words (using the sourced word swapper) so detection methods run on the flipped text
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SWAPPER = Path(__file__).resolve().parent / "sourcedWordSwapper.py"
TOKEN = re.compile(r"\w+(?:'\w+)*|[^\w\s]")
BATCH_SIZE = 1000

def to_conll(batch):
    lines = []
    for index, tokens in enumerate(batch):
        lines.append(f"#begin document (d{index}); part 000")
        lines += [f"d{index}\t000\t{position}\t{token}\tNN\t-" for position, token in enumerate(tokens)]
        lines.append("")
    lines.append("#begin document (end); part 000")
    return "\n".join(lines) + "\n"

def swap_batch(batch):
    with tempfile.NamedTemporaryFile("w", suffix=".conll", delete=False, encoding="utf-8") as conll:
        conll.write(to_conll(batch))
    try:
        completed = subprocess.run(
            [sys.executable, str(SWAPPER), conll.name],
            capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    finally:
        os.unlink(conll.name)

    if completed.returncode != 0:
        raise RuntimeError(f"Word swapper failed:\n{completed.stderr}")

    swapped, current = [], None
    for line in completed.stdout.splitlines():
        if line.startswith("#begin"):
            current = []
            swapped.append(current)
        elif line.strip() and not line.startswith("#end"):
            current.append(line.split()[3])
    return swapped

def rebuild(text, matches, swapped):
    pieces, last = [], 0
    for match, word in zip(matches, swapped):
        pieces += [text[last:match.start()], word]
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)

def transform(texts):
    matches = [list(TOKEN.finditer(text)) for text in texts]
    indexes = [index for index, found in enumerate(matches) if found]
    result = list(texts)

    for start in range(0, len(indexes), BATCH_SIZE):
        chunk = indexes[start:start + BATCH_SIZE]
        swapped_batch = swap_batch([[match.group() for match in matches[index]] for index in chunk])
        if len(swapped_batch) != len(chunk):
            raise RuntimeError("Word swapper returned a different number of documents than it was given")

        for index, swapped in zip(chunk, swapped_batch):
            if len(swapped) != len(matches[index]):
                raise RuntimeError("Word swapper returned a different number of words than it was given")
            result[index] = rebuild(texts[index], matches[index], swapped)

    return result
