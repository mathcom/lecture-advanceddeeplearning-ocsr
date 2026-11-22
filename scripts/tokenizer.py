"""SMILES 토크나이저.

SMILES 문자열을 원자 단위 토큰으로 나누고, 토큰과 번호를 서로 바꾼다.
예: "C[C@H](N)Cl" -> ["C", "[C@H]", "(", "N", ")", "Cl"]
"""

import json
import re
from pathlib import Path
from typing import Iterable

# 대괄호 원자([nH], [O-] 등)는 통째로 한 토큰, Cl과 Br도 한 토큰이다.
SMILES_TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|Br|Cl|%\d{2}|[BCNOSPFIbcnosp]|[=#\-+\\/().@:~*$]|\d"
)

# 특수 토큰의 번호는 항상 이 순서(0, 1, 2, 3)로 고정한다.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


class SmilesTokenizer:
    """원자 단위 SMILES 토크나이저.

    Args:
        tokens: 번호 순서대로 나열한 전체 토큰 목록(특수 토큰 4개로 시작).
        max_length: <bos>, <eos>를 포함한 최대 토큰 수.
    """

    def __init__(self, tokens: list[str], max_length: int):
        if tokens[:4] != SPECIAL_TOKENS:
            raise ValueError(f"토큰 목록은 {SPECIAL_TOKENS}로 시작해야 합니다.")
        self.tokens = list(tokens)
        self.max_length = max_length
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, smiles_iterable: Iterable[str], max_length: int) -> "SmilesTokenizer":
        """SMILES들에 나온 모든 토큰으로 어휘를 만든다.

        Args:
            smiles_iterable: SMILES 문자열들.
            max_length: <bos>, <eos>를 포함한 최대 토큰 수.
        Returns:
            특수 토큰 4개 뒤에 나머지 토큰을 문자열 정렬 순으로 둔 토크나이저.
        """
        found = set()
        for smiles in smiles_iterable:
            found.update(split_smiles(smiles))
        return cls(SPECIAL_TOKENS + sorted(found), max_length)

    @property
    def vocab_size(self) -> int:
        """특수 토큰을 포함한 어휘 크기."""
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def tokenize(self, smiles: str) -> list[str]:
        """SMILES를 토큰 문자열의 목록으로 나눈다."""
        return split_smiles(smiles)

    def encode(self, smiles: str) -> list[int]:
        """SMILES를 `[<bos>] + 토큰 + [<eos>]` 번호 목록으로 바꾼다.

        어휘에 없는 토큰은 <unk>가 된다. 길이가 max_length를 넘으면 ValueError.
        """
        body = [self.token_to_id.get(token, UNK_ID) for token in self.tokenize(smiles)]
        ids = [BOS_ID] + body + [EOS_ID]
        if len(ids) > self.max_length:
            raise ValueError(
                f"토큰 수 {len(ids)}개가 max_length {self.max_length}을 넘습니다: {smiles}"
            )
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """번호 목록을 SMILES로 되돌린다.

        <bos>는 건너뛰고, 첫 <eos>에서 멈추며, <pad>와 <unk>는 버린다.
        """
        pieces = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == EOS_ID:
                break
            if token_id in (PAD_ID, BOS_ID, UNK_ID):
                continue
            pieces.append(self.tokens[token_id])
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        """토큰 목록과 max_length를 JSON 파일로 저장한다."""
        content = {"max_length": self.max_length, "tokens": self.tokens}
        text = json.dumps(content, ensure_ascii=False, indent=2)
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        """`save`로 저장한 파일에서 토크나이저를 복원한다."""
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(content["tokens"], content["max_length"])


def split_smiles(smiles: str) -> list[str]:
    """SMILES를 토큰으로 나눈다. 읽지 못하는 문자가 있으면 ValueError."""
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"토큰으로 나눌 수 없는 문자가 있습니다: {smiles!r}")
    return tokens
