"""Create a tiny SYNTHETIC dataset shaped like a speech corpus, for smoke tests only.

The audio is noise/tones and the "Dholuo" sentences are placeholders; nothing here is
real data and nothing produced from it is a result. Deliberate defects are injected so
the audit notebook's checks can be exercised: a missing file, a corrupt file, a
duplicated row, an empty transcript, a bracketed tag, curly apostrophes, a digit and an
English token.

usage: python tests/make_fake_dataset.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PROMPTS = [
    "an gi nyithindo ariyo", "dala ber ahinya", "ng'ato ok bi kawuono", "chiemo osechiek",
    "wadhi e chiro kiny", "piny ochido ahinya", "japuonj osebiro e school", "nyithindo tugo e pap",
    "kodh ochwe nyoro", "wan gi dhok apar 10", "Ng’ama nono ok ong'eyo", "en ja kisumu",
    "[noise] jo dala ne ochiemo", "mos mos ema iwuothoe", "ndiga mara obedo maber",
    "koth biro chieng ma lero", "", "wuod ng'ato ne obiro", "nyar migori dhi skul", "an gi kech",
]


def main(out: Path):
    rng = np.random.default_rng(0)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows = []
    speakers = [(f"spk{i:02d}", "Milambo" if i < 6 else "Nyanduat",
                 "female" if i % 2 else "male", ["19-29", "30-39", "40-49"][i % 3]) for i in range(10)]
    domains = ["health", "agriculture", "education", "general"]
    k = 0
    for spk, dia, gen, age in speakers:
        for j in range(14):
            text = PROMPTS[(j * 3 + int(spk[-2:])) % len(PROMPTS)]
            fn = f"{spk}_{j:03d}.wav"
            n_words = max(1, len(text.split()))
            dur = 0.35 * n_words + 0.6 + rng.uniform(0, 0.4)
            sr = 22050 if k % 2 else 48000
            t = np.arange(int(dur * sr)) / sr
            wav = 0.1 * np.sin(2 * np.pi * (180 + 40 * rng.random()) * t) + 0.02 * rng.standard_normal(len(t))
            wav[: int(0.2 * sr)] = 0.0005 * rng.standard_normal(int(0.2 * sr))  # leading silence
            sf.write(clips / fn, wav.astype(np.float32), sr)
            rows.append({"recording_id": f"r{k:05d}", "audio": f"clips/{fn}", "transcription": text,
                         "speaker_id": spk, "dialect": dia, "gender": gen, "age_group": age,
                         "domain": domains[j % 4]})
            k += 1
    rows.append({**rows[3], "recording_id": "r_missing", "audio": "clips/does_not_exist.wav"})
    (clips / "corrupt.wav").write_bytes(b"RIFF....not really a wav file")
    rows.append({**rows[5], "recording_id": "r_corrupt", "audio": "clips/corrupt.wav"})
    rows.append(dict(rows[7]))  # exact duplicate row
    pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)
    (out / "README.txt").write_text("synthetic smoke-test data\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
