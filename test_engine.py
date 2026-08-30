import soundfile as sf
from qwen_stt_service import Qwen3ASREngine
from config import config

print("Initializing Qwen3ASREngine...")
engine = Qwen3ASREngine(
    model_path=config.asr.model_path,
    language="Hindi",
    use_quantization=config.asr.use_quantization,
    num_threads=config.asr.num_threads,
)
data, sr = sf.read("test_cartesia.wav")
text, lang, lat = engine.transcribe_pcm(data, sr)
print(f"Decoded: text='{text}', lang='{lang}', lat={lat:.1f}ms")
assert text != ""
print("SUCCESS: transcribe_pcm test passed!")
