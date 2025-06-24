# Text Splitter 설정 함수들
from langchain.text_splitter import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
import MeCab

def setup_recursive_text_splitter(chunk_size=1000, chunk_overlap=200):
    """
    Recursive Character Text Splitter 설정
    - 가장 일반적이고 권장되는 분할기
    - 문단, 문장, 단어 순으로 분할 시도
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

def setup_character_text_splitter(chunk_size=1000, chunk_overlap=200, separator="\n\n"):
    """
    Character Text Splitter 설정
    - 특정 구분자로 분할
    """
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator
    )

def setup_token_text_splitter(chunk_size=512, chunk_overlap=50):
    """
    Token Text Splitter 설정
    - 토큰 단위로 분할 (임베딩 모델의 최대 길이 고려)
    """
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


class MecabTextSplitter(TextSplitter):
    def __init__(self, chunk_size=300, chunk_overlap=30):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.tokenizer = MeCab.Tagger("-Owakati")  # 띄어쓰기 기반 출력

    def split_text(self, text: str):
        tokens = self.tokenizer.parse(text).strip().split()
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(" ".join(chunk))
        return chunks