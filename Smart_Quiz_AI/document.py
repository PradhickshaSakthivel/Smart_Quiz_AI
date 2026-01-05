import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)

class Document:
    def extract_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_chunks(self, text, size=500):
        sentences = sent_tokenize(text)
        chunks, current = [], ""

        for s in sentences:
            if len(current) + len(s) < size:
                current += s + " "
            else:
                chunks.append(current)
                current = s + " "

        if current:
            chunks.append(current)

        return chunks
 