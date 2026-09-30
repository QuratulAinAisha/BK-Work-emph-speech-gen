"""Full LLM-free speech architecture. / LLM 없는 전체 음성 구조."""

from .config import SpeechConfig
from .system import EmpatheticSpeechSystem

__all__ = ["SpeechConfig", "EmpatheticSpeechSystem"]
