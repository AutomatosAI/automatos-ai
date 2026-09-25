"""v2 mix: Afro-house bed (ducked under voice) + voice-over + SFX, normalised to -14 LUFS."""
import subprocess, sys
C = sys.argv[1]
DUR = 40.0
vo = [("l01", 0.30), ("l02", 2.40), ("l03", 3.95), ("l04", 9.30), ("l05", 12.30), ("l06", 17.40),
      ("l07", 20.90), ("l08", 23.70), ("l09", 27.60), ("l10", 32.40)]
sfx = [("switch_002", 3.28, 0.70), ("impactSoft_medium_001", 3.36, 0.40), ("impactBell_heavy_003", 4.37, 0.35),
       ("click1", 10.44, 0.50), ("card-fan-1", 12.20, 0.30), ("card-slide-1", 13.92, 0.45), ("card-slide-2", 14.56, 0.40),
       ("card-slide-4", 15.10, 0.40), ("card-slide-1", 15.65, 0.45), ("mouseclick1", 19.05, 0.60), ("drop_002", 19.30, 0.40),
       ("drop_001", 23.75, 0.35), ("drop_001", 25.60, 0.30), ("impactBell_heavy_000", 32.45, 0.45), ("select_008", 34.40, 0.30)]
fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
args = ["ffmpeg", "-v", "error", "-y", "-i", f"{C}/assets/music/dadada-excerpt.wav"]
args += sum([["-i", f"{C}/assets/vo/{k}.wav"] for k, _ in vo], [])
args += sum([["-i", f"{C}/assets/sfx/{k}.ogg"] for k, _, _ in sfx], [])
f = [f"[0:a]{fmt},afade=t=in:st=0:d=0.02,afade=t=out:st=38.3:d=1.7,atrim=0:{DUR}[m]"]
for i, (k, t) in enumerate(vo, start=1):
    ms = int(round(t * 1000)); f.append(f"[{i}:a]{fmt},adelay={ms}|{ms}[v{i}]")
f.append("".join(f"[v{i}]" for i in range(1, len(vo) + 1)) + f"amix=inputs={len(vo)}:normalize=0:duration=longest,apad=whole_dur={DUR},atrim=0:{DUR}[vo]")
f.append("[vo]asplit=2[vomix][vokey]")
f.append("[m][vokey]sidechaincompress=threshold=0.015:ratio=10:attack=10:release=420:makeup=1[mduck]")
base = 1 + len(vo)
for j, (k, t, vol) in enumerate(sfx):
    ms = int(round(t * 1000)); f.append(f"[{base + j}:a]{fmt},volume={vol},adelay={ms}|{ms}[s{j}]")
f.append("".join(f"[s{j}]" for j in range(len(sfx))) + f"amix=inputs={len(sfx)}:normalize=0:duration=longest,apad=whole_dur={DUR},atrim=0:{DUR}[sx]")
f.append("[mduck]volume=0.6[mq]")
f.append(f"[mq][vomix][sx]amix=inputs=3:normalize=0:duration=longest,atrim=0:{DUR},loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,{fmt}[out]")
args += ["-filter_complex", ";".join(f), "-map", "[out]", "-c:a", "pcm_s16le", f"{C}/assets/audio/mix.wav"]
subprocess.run(args, check=True); print("ok")
