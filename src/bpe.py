# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        """
        1. 특수 토큰 4개를 등록한다.
        SPECIAL_IDS.items() 에는 다음과 같이 들어있다.
            ('<pad>', 0)
            ('<unk>', 1)
            ('<bos>', 2)
            ('<eos>', 3)
        
        해당 token과 token_id를 id_to_token 배열과 token_to_id 배열에 등록한다.
        """
        for special_token, special_token_id in SPECIAL_IDS.items():
            self.id_to_token[special_token_id] = special_token
            self.token_to_id[special_token] = special_token_id

        """
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록한다.
        BYTE_OFFSET(=4) 만큼 더해준 값을 token_id 로 설정한다.
        bytes([byte]) => byte로 들어온 값(4 ~ 259)를 bytes()를 사용해서 정수를 바이트 객체로 변환해준다.
        ex) bytes([72, 101, 108, 108, 111]) => b'Hello'
        """
        for byte in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte
            byte_token = bytes([byte])

            self.id_to_token[token_id] = byte_token
            self.token_to_id[byte_token] = token_id

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """

        """
        정리한 BPE train 과정
        1. corpus를 byte token ID 목록으로 바꾼다.
        2. 인접한 token pair의 빈도를 센다.
        3. 가장 자주 등장한 pair를 고른다 (pair 빈도가 같은 경우에는 설정한 tie-break 규칙에 따라 처리한다)
        4. 그 pair에 새로운 token ID를 부여한다.
        5. corpus 안의 해당 pair를 새 token으로 교체한다.
        6. 목표 vocab_size에 도달할 때까지 반복한다. 2번으로 루프

        """
        self._init_special_tokens()
        token_id_list = []
        # UTF-8 byte를 기본 byte token ID로 변환
        for byte in corpus.encode("utf-8"):
            token_id_list.append(byte + BYTE_OFFSET)

        """
        vocab_size = 토큰을 최대 몇개까지 만들지 제한하는 값(여기서는 3000으로 선언)
        id_to_token의 길이가 vocab_size와 같아지면? merge 루프를 종료한다.
        """
        while len(self.id_to_token) < self.vocab_size:
            """
            이웃한 token pair의 빈도를 센다.
            """
            # 빈도수를 기록 할 dict 선언
            pair_count = {}

            for i in range(len(token_id_list) - 1):
                pair = (token_id_list[i], token_id_list[i + 1])
                if pair not in pair_count:
                    pair_count[pair] = 0
                pair_count[pair] += 1

            """
            가장 자주 등장한 pair를 고른다.
            """
            freq_pair = None
            freq_count = 0

            for pair, count in pair_count.items():
                if count > freq_count:
                    freq_pair = pair
                    freq_count = count

            if freq_count < 2:
                break
            """
            그 pair에 새로운 token ID를 부여한다.
            """
            # 새로운 token ID를 어떻게 받아오지.. len?
            new_token_id = len(self.id_to_token)
            self.id_to_token[new_token_id] = freq_pair
            self.token_to_id[freq_pair] = new_token_id
            self.merges.append(freq_pair)

            """
            token_list에서 해당 pair를 새 token으로 교체한다.
            """
            # token_id 리스트를 또 만들어야할까? 너무 비용이 클 거 같은데
            # 기존 token_id 리스트에서 해당 pair 만 교체하는 방법이 있을까?
            new_token_id_list = []
            i = 0
            while i < len(token_id_list):
                if i < len(token_id_list) - 1 and token_id_list[i] == freq_pair[0] and token_id_list[i + 1] == freq_pair[1]:
                    # new_token_id_list에 어떤걸 append?
                    # 새롭게 생성한 token id를 append 하면 될 것 같다.
                    new_token_id_list.append(new_token_id)
                    i += 2
                else:
                    # freq_pair가 아닌 token은 그대로 copy
                    new_token_id_list.append(token_id_list[i])
                    i += 1

            token_id_list = new_token_id_list

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        token_id_list = []

        for byte in text.encode("utf-8"):
            token_id_list.append(byte + BYTE_OFFSET)

        for pair in self.merges:
            new_token_id = self.token_to_id[pair]

            new_token_id_list = []
            i = 0

            while i < len(token_id_list):
                if i < len(token_id_list) - 1 and token_id_list[i] == pair[0] and token_id_list[i + 1] == pair[1]:
                    new_token_id_list.append(new_token_id)
                    i += 2
                else:
                    new_token_id_list.append(token_id_list[i])
                    i += 1

            token_id_list = new_token_id_list

        if add_bos_eos:
            token_id_list = [self.get_bos_id()] + token_id_list + [self.get_eos_id()]

        return token_id_list

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        # 입력값으로 받은 encode 된 list
        # 일단 펼친다.
        # for token_id in ids:
