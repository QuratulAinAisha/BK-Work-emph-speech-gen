"""Load model families on demand. / 필요한 모델만 불러옵니다."""

from importlib import import_module


# Keep legacy exports without optional imports. / 기존 이름과 선택적 의존성을 보존합니다.
_EXPORTS = {
    "AudioEmbedder": (".audio_model.audio_embedder", "AudioEmbedder"),
    "AutoencoderRNN_VAE_v1": (".diffusion.rnn", "AutoencoderRNN_VAE_v1"),
    "AutoencoderRNN_VAE_v2": (".diffusion.rnn", "AutoencoderRNN_VAE_v2"),
    "PriorLatentMatcher": (".diffusion.matchers", "PriorLatentMatcher"),
    "DecoderLatentMatcher": (".diffusion.matchers", "DecoderLatentMatcher"),
    "LatentMatcher": (".diffusion.matchers", "LatentMatcher"),
    "Transformer": (".person_specific.PersonSpecificEncoder", "Transformer"),
    "MainNetUnified": (".modifier.network", "MainNetUnified"),
    "AffectiveResponseTransport": (".affective_response_transport", "AffectiveResponseTransport"),
    "AffectiveTransportConfig": (".affective_response_transport", "AffectiveTransportConfig"),
    "SBEWithAffectiveTransport": (".affective_pipeline", "SBEWithAffectiveTransport"),
    "SpeechConfig": (".full_speech.config", "SpeechConfig"),
    "EmpatheticSpeechSystem": (".full_speech.system", "EmpatheticSpeechSystem"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value
