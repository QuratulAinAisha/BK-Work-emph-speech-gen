"""Connect existing modules 1-2 to module 3. / 기존 모듈 1-2를 모듈 3에 연결합니다."""

from torch import Tensor, nn

from .affective_response_transport import (
    AffectiveResponseTransport, checked_lengths, clean_padding, valid_mask,
)
from .speaker_behavior_encoder import SpeakerBehaviorEncoder


class SBEWithAffectiveTransport(nn.Module):
    """Person-A features -> c -> a_B*. / A 특징 -> 문맥 -> B 목표 정서.

    No decoder or training loop is added. / 디코더나 학습 루프는 추가하지 않습니다.
    """

    def __init__(self, sbe: SpeakerBehaviorEncoder,
                 transport: AffectiveResponseTransport | None = None):
        super().__init__()
        self.sbe = sbe
        self.transport = transport if transport is not None else AffectiveResponseTransport()

    def forward(self, mel: Tensor, dmm: Tensor, au: Tensor,
                mel_len: Tensor | None = None, dmm_len: Tensor | None = None,
                au_len: Tensor | None = None) -> dict[str, Tensor]:
        mel_len = checked_lengths(mel, mel_len, "mel")
        dmm_len = checked_lengths(dmm, dmm_len, "dmm")
        au_len = checked_lengths(au, au_len, "au")
        if mel.shape[0] != dmm.shape[0] or mel.shape[0] != au.shape[0]:
            raise ValueError("mel, dmm, and au batch sizes must match")
        if not (mel.device == dmm.device == au.device and mel.dtype == dmm.dtype == au.dtype):
            raise ValueError("mel, dmm, and au must have the same device and dtype")
        # Remove padding before any branch sees it. / 각 인코더에 넣기 전에 패딩을 제거합니다.
        mel = clean_padding(mel, valid_mask(mel_len, mel.shape[1]), "mel")
        dmm = clean_padding(dmm, valid_mask(dmm_len, dmm.shape[1]), "dmm")
        au = clean_padding(au, valid_mask(au_len, au.shape[1]), "au")
        context, context_mask = self.sbe(
            mel, dmm, au, mel_len=mel_len, dmm_len=dmm_len, au_len=au_len,
            return_mask=True,
        )
        affect = self.transport(context, au, context_mask, au_len)
        # Handoff to future modules 4, 5, and 6. / 향후 모듈 4, 5, 6에 전달합니다.
        return {
            "context": context,
            "context_mask": context_mask,
            "affect": affect.trajectory,
            "affect_mask": affect.mask,
            "affect_lengths": affect.lengths,
            "affect_summary": affect.mean_pool(),
        }
