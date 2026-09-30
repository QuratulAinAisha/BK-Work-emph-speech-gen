"""Load supported response checkpoints strictly. / 지원 응답 체크포인트를 엄격히 읽습니다."""

import torch
from .quality import QualityConfig, QualitySpeechSystem


def load_response_model(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    architecture = payload.get('architecture')
    config = QualityConfig(**payload['config'])
    if architecture == 'llm_free_speech_quality_v2':
        model = QualitySpeechSystem(config)
    elif architecture == 'llm_free_speech_units_v1':
        from .units import UnitSpeechSystem
        model = UnitSpeechSystem(config, payload['state_dict']['semantic_planner.codebook.centers'])
    else:
        raise ValueError(f'Unsupported response architecture: {architecture}')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.current_epoch = payload.get('current_epoch', 0)
    return model.eval(), payload
