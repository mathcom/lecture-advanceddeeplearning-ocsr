"""OCSR 모델: ViT 인코더와 GPT 디코더.

전체 구조: 이미지 -> ViT 인코더 -> 패치 표현 -> GPT 디코더(cross-attention) -> 다음 토큰 분포

shape 주석의 기호
    B: 배치 크기, T: 토큰열 길이, N: 인코더 출력 토큰 수, D: 은닉 차원, V: 어휘 크기
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.tokenizer import PAD_ID, SmilesTokenizer


# ---------------------------------------------------------------------------
# 공통 부품
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    context를 주지 않으면 self-attention, 주면 cross-attention이다.

    Args:
        dim: 은닉 차원 D. num_heads로 나누어떨어져야 한다.
        num_heads: 헤드 수 H.
        dropout: attention 가중치에 적용할 dropout 비율.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim})은 num_heads({num_heads})로 나누어떨어져야 합니다.")
        self.num_heads = num_heads
        self.dropout = dropout
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, is_causal: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: query를 만들 표현 (B, T, D)
            context: key와 value를 만들 표현 (B, S, D). None이면 x를 쓴다.
            is_causal: True이면 각 위치가 자신과 그 앞만 참조한다.
        Returns:
            (B, T, D)
        """
        if context is None:
            context = x
        q = self.split_heads(self.query(x))        # (B, H, T, D/H)
        k = self.split_heads(self.key(context))    # (B, H, S, D/H)
        v = self.split_heads(self.value(context))  # (B, H, S, D/H)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )                                          # (B, H, T, D/H)

        B, H, T, head_dim = out.shape
        out = out.transpose(1, 2)                  # (B, T, H, D/H)
        out = out.reshape(B, T, H * head_dim)      # (B, T, D)
        return self.out(out)                       # (B, T, D)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, D/H)"""
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, D // self.num_heads)  # (B, T, H, D/H)
        return x.transpose(1, 2)                                  # (B, H, T, D/H)


class MLP(nn.Module):
    """위치마다 따로 적용하는 2층 MLP. 은닉층의 크기는 dim * mlp_ratio이다."""

    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)"""
        x = F.gelu(self.fc1(x))  # (B, T, D * mlp_ratio)
        return self.fc2(x)       # (B, T, D)


def init_weights(module: nn.Module) -> None:
    """Linear와 Embedding을 초기화한다. `model.apply(init_weights)` 형태로 쓴다.

    가중치는 표준편차 0.02의 truncated normal, bias는 0이다. 이렇게 하면 처음의 출력
    분포가 거의 균등해져 손실이 ln(어휘 크기) 근처에서 출발한다.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# 인코더
# ---------------------------------------------------------------------------

class TimmEncoder(nn.Module):
    """timm의 ViT를 인코더로 쓴다.

    Args:
        name: timm 모델 이름.
        pretrained: 사전학습 가중치 사용 여부.
        image_size: 입력 이미지의 한 변. 사전학습 해상도와 다르면 위치 임베딩이 보간된다.
        drop_path_rate: stochastic depth 비율.
    """

    def __init__(self, name: str, pretrained: bool, image_size: int, drop_path_rate: float):
        super().__init__()
        try:
            self.vit = timm.create_model(
                name,
                pretrained=pretrained,
                img_size=image_size,
                num_classes=0,
                drop_path_rate=drop_path_rate,
            )
        except Exception as error:
            # 무작위 초기화로 조용히 넘어가지 않는다.
            raise RuntimeError(
                f"timm 인코더 {name}을 만들지 못했습니다(pretrained={pretrained}). "
                "사전학습 가중치를 내려받을 수 있는지(인터넷 연결) 확인하세요."
            ) from error
        self.dim = self.vit.num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, D_enc). N에는 CLS 토큰이 포함된다."""
        return self.vit.forward_features(images)  # (B, N, D_enc)


class EncoderBlock(nn.Module):
    """ViT의 pre-LN Transformer 블록: self-attention -> MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        x = x + self.dropout(self.attn(self.norm1(x)))  # (B, N, D)
        x = x + self.dropout(self.mlp(self.norm2(x)))   # (B, N, D)
        return x


class ScratchViTEncoder(nn.Module):
    """직접 구현한 ViT 인코더. CLS 토큰은 두지 않는다.

    Args:
        image_size: 입력 이미지의 한 변. patch_size로 나누어떨어져야 한다.
        patch_size: 패치의 한 변.
        dim: 은닉 차원.
        depth: 블록 수.
        num_heads: 헤드 수.
        mlp_ratio: MLP 확장 비율.
        dropout: dropout 비율.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size는 patch_size로 나누어떨어져야 합니다.")
        self.dim = dim
        num_patches = (image_size // patch_size) ** 2

        # 커널과 stride가 patch_size인 Conv2d는 패치마다 같은 선형 변환을 적용하는 것과 같다.
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.apply(init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, D). N은 패치 수이다."""
        x = self.patch_embed(images)              # (B, D, H/P, W/P)
        x = x.flatten(2)                          # (B, D, N)
        x = x.transpose(1, 2)                     # (B, N, D)
        x = self.dropout(x + self.pos_embed)      # (B, N, D)
        for block in self.blocks:
            x = block(x)                          # (B, N, D)
        return self.norm(x)                       # (B, N, D)


def build_encoder(config: dict, load_pretrained: bool = True) -> nn.Module:
    """설정에 따라 인코더를 만든다. 인코더의 출력은 (B, N, D_enc)이다.

    Args:
        config: 설정.
        load_pretrained: False이면 encoder.pretrained와 무관하게 사전학습 가중치를
            내려받지 않는다.
    """
    encoder_config = config["encoder"]
    image_size = config["image"]["size"]
    if encoder_config["type"] == "timm":
        return TimmEncoder(
            name=encoder_config["name"],
            pretrained=encoder_config["pretrained"] and load_pretrained,
            image_size=image_size,
            drop_path_rate=encoder_config["drop_path_rate"],
        )
    return ScratchViTEncoder(
        image_size=image_size,
        patch_size=encoder_config["patch_size"],
        dim=encoder_config["dim"],
        depth=encoder_config["depth"],
        num_heads=encoder_config["num_heads"],
        mlp_ratio=encoder_config["mlp_ratio"],
        dropout=encoder_config["dropout"],
    )


def image_normalization(config: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """이미지 정규화에 쓸 (mean, std)를 돌려준다. 모델을 만들지 않고 얻는다."""
    if config["encoder"]["type"] == "timm":
        pretrained_cfg = timm.get_pretrained_cfg(config["encoder"]["name"])
        return pretrained_cfg.mean, pretrained_cfg.std
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# 디코더
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """GPT의 pre-LN 블록: masked self-attention -> cross-attention -> MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 토큰 표현 (B, T, D)
            memory: 인코더 출력 (B, N, D)
        Returns:
            (B, T, D)
        """
        # causal mask: 각 토큰은 자신과 그 앞의 토큰만 본다.
        x = x + self.dropout(self.self_attn(self.norm1(x), is_causal=True))     # (B, T, D)
        # 각 토큰이 이미지 패치들을 참조한다.
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=memory))    # (B, T, D)
        x = x + self.dropout(self.mlp(self.norm3(x)))                           # (B, T, D)
        return x


class GPTDecoder(nn.Module):
    """cross-attention이 있는 GPT 디코더.

    Args:
        vocab_size: 어휘 크기 V.
        max_length: 최대 토큰 수(위치 임베딩의 길이).
        dim: 은닉 차원 D.
        depth: 블록 수.
        num_heads: 헤드 수.
        mlp_ratio: MLP 확장 비율.
        dropout: dropout 비율.
        encoder_dim: 인코더 출력 차원 D_enc. dim과 다르면 선형층으로 맞춘다.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        encoder_dim: int,
    ):
        super().__init__()
        self.max_length = max_length
        self.memory_proj = nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.apply(init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # weight tying: 출력층과 토큰 임베딩이 같은 가중치를 쓴다.
        self.head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: 토큰 번호 (B, T)
            memory: 인코더 출력 (B, N, D_enc)
        Returns:
            위치마다 다음 토큰의 logits (B, T, V)
        """
        T = input_ids.shape[1]
        if T > self.max_length:
            raise ValueError(f"토큰 수 {T}개가 max_length {self.max_length}을 넘습니다.")

        memory = self.memory_proj(memory)           # (B, N, D)
        x = self.token_embed(input_ids)             # (B, T, D)
        x = x + self.pos_embed[:, :T]               # (B, T, D)
        x = self.dropout(x)                         # (B, T, D)
        for block in self.blocks:
            x = block(x, memory)                    # (B, T, D)
        x = self.norm(x)                            # (B, T, D)
        return self.head(x)                         # (B, T, V)


# ---------------------------------------------------------------------------
# 전체 모델
# ---------------------------------------------------------------------------

class OCSRModel(nn.Module):
    """이미지를 읽어 SMILES 토큰열을 만드는 모델.

    Args:
        encoder: (B, 3, H, W) -> (B, N, D_enc)
        decoder: (B, T), (B, N, D_enc) -> (B, T, V)
    """

    def __init__(self, encoder: nn.Module, decoder: GPTDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """teacher forcing으로 다음 토큰의 logits를 계산한다.

        Args:
            images: (B, 3, H, W)
            input_ids: 정답 토큰열 (B, T)
        Returns:
            (B, T, V)
        """
        memory = self.encoder(images)          # (B, N, D_enc)
        return self.decoder(input_ids, memory)  # (B, T, V)

    @torch.no_grad()
    def generate(
        self, images: torch.Tensor, bos_id: int, eos_id: int, max_new_tokens: int
    ) -> torch.Tensor:
        """greedy 디코딩으로 토큰열을 생성한다. 미리 `model.eval()`을 호출해 두어야 한다.

        Args:
            images: (B, 3, H, W)
            bos_id: 시작 토큰 번호.
            eos_id: 끝 토큰 번호.
            max_new_tokens: <bos> 뒤에 생성할 최대 토큰 수.
        Returns:
            <bos>로 시작하는 토큰열 (B, ≤max_new_tokens+1). <eos> 뒤는 <pad>이다.
        """
        # <bos>를 포함한 길이가 위치 임베딩의 길이를 넘지 않게 한다.
        max_new_tokens = min(max_new_tokens, self.decoder.max_length - 1)

        B = images.shape[0]
        memory = self.encoder(images)  # (B, N, D_enc), 인코더는 한 번만 실행한다.
        ids = torch.full((B, 1), bos_id, dtype=torch.long, device=images.device)  # (B, 1)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)         # (B,)

        for _ in range(max_new_tokens):
            logits = self.decoder(ids, memory)                    # (B, T, V)
            next_ids = logits[:, -1].argmax(dim=-1)               # (B,)
            next_ids = next_ids.masked_fill(finished, PAD_ID)     # (B,) 끝난 표본은 <pad>
            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)  # (B, T+1)
            finished = finished | (next_ids == eos_id)            # (B,)
            if finished.all():
                break
        return ids


def build_model(config: dict, tokenizer: SmilesTokenizer, load_pretrained: bool = True) -> OCSRModel:
    """설정에 따라 전체 모델을 만든다.

    Args:
        config: 설정.
        tokenizer: 어휘 크기와 최대 길이를 얻을 토크나이저.
        load_pretrained: False이면 사전학습 가중치를 내려받지 않는다. 체크포인트로
            가중치를 덮어쓰는 경우(재개, 추론)에 쓴다.
    """
    encoder = build_encoder(config, load_pretrained)
    decoder_config = config["decoder"]
    decoder = GPTDecoder(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
        dim=decoder_config["dim"],
        depth=decoder_config["depth"],
        num_heads=decoder_config["num_heads"],
        mlp_ratio=decoder_config["mlp_ratio"],
        dropout=decoder_config["dropout"],
        encoder_dim=encoder.dim,
    )
    return OCSRModel(encoder, decoder)


def compute_loss(
    logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
) -> torch.Tensor:
    """<pad>를 무시하는 cross-entropy 손실(토큰당 평균).

    Args:
        logits: (B, T, V)
        targets: 각 위치에서 맞혀야 할 다음 토큰 번호 (B, T)
        label_smoothing: label smoothing 비율.
    Returns:
        스칼라 텐서.
    """
    return F.cross_entropy(
        logits.flatten(0, 1),  # (B*T, V)
        targets.flatten(),     # (B*T,)
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
    )


def count_parameters(model: OCSRModel) -> dict:
    """인코더, 디코더, 전체의 파라미터 수를 센다."""
    encoder = sum(p.numel() for p in model.encoder.parameters())
    decoder = sum(p.numel() for p in model.decoder.parameters())
    return {"encoder": encoder, "decoder": decoder, "total": encoder + decoder}
