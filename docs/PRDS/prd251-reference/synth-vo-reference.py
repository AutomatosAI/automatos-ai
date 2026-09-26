import json, os, sys, soundfile as sf
import kokoro_onnx
from kokoro_onnx.config import EspeakConfig
import espeakng_loader
home = os.path.expanduser("~/.cache/hyperframes")
cfg = EspeakConfig(lib_path=espeakng_loader.get_library_path(), data_path=os.path.join(home, "espeak-ng-data"))
m = kokoro_onnx.Kokoro(os.path.join(home, "tts/models/kokoro-v1.0.onnx"), os.path.join(home, "tts/voices/voices-v1.0.bin"), espeak_config=cfg)
lines = json.load(open(sys.argv[1]))
voice = sys.argv[2]; speed = float(sys.argv[3]); outdir = sys.argv[4]
os.makedirs(outdir, exist_ok=True)
res = {}
for key, text in lines.items():
    s, sr = m.create(text, voice=voice, speed=speed, lang="en-us")
    p = os.path.join(outdir, f"{key}.wav"); sf.write(p, s, sr)
    res[key] = round(len(s)/sr, 3)
print(json.dumps(res, indent=1))
