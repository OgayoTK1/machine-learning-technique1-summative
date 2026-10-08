"""Dholuo speech-to-text demo (Gradio). Serves the model selected in notebook 06.

Run locally:   MODEL_DIR=path/to/results/<run>/model python app/app.py
Hugging Face Spaces: copy app/, src/ and the model folder; set MODEL_DIR in the Space settings.
"""
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dholuo_asr.inference import Transcriber  # noqa: E402

MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT / "model"))
MODEL_NAME = os.environ.get("MODEL_NAME", Path(MODEL_DIR).parent.name or "model")
TEST_WER = os.environ.get("TEST_WER", "see report")   # fill from evaluation/final_summary.json
MAX_SECONDS = 30

asr = Transcriber(MODEL_DIR)

ABOUT = f"""
### What this does
Transcribes **short Dholuo speech** (Milambo and Nyanduat dialects) into Dholuo text.
Model: `{MODEL_NAME}`, fine-tuned on the DhoNam: Dholuo Speech dataset (Maseno University MCAAI & Strathmore CIPIT,
via Mozilla Data Collective, NOODL-1.0). Test WER on unseen speakers: **{TEST_WER}**.

### What it does NOT do
* It does not translate, understand meaning or identify speakers.
* It is not reliable for other languages, heavy code-switching, noisy recordings, or speech longer than {MAX_SECONDS} s.
* Transcripts **will contain errors**. Do not use it for medical, legal, financial or any other high-stakes record.

### Expected input
A clear recording of one speaker saying a sentence in Dholuo (microphone or file; any sample rate, converted to 16 kHz mono).
Audio is processed in memory and **not stored**.
"""


def run(audio):
    if audio is None:
        return "", "Record or upload audio first."
    sr, wav = audio
    import numpy as np
    wav = wav.astype("float32")
    if np.abs(wav).max() > 1.0:                 # int16 from the microphone component
        wav = wav / 32768.0
    dur = len(wav) / sr
    if dur > MAX_SECONDS:
        return "", f"Audio is {dur:.0f} s; please keep it under {MAX_SECONDS} s."
    r = asr.transcribe(wav, sr)
    note = (f"Model confidence: {r['confidence']:.0%} (the model's own certainty, not a guarantee of accuracy) "
            f"| duration {r['duration_s']:.1f} s")
    if r["confidence"] < 0.6:
        note += " | LOW confidence: treat this transcript with extra caution."
    return r["text"], note


with gr.Blocks(title="Dholuo speech-to-text") as demo:
    gr.Markdown("# Dholuo speech-to-text (research demo)")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Dholuo audio")
            btn = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            text = gr.Textbox(label="Transcript", lines=4)
            info = gr.Markdown()
    btn.click(run, inputs=audio, outputs=[text, info])
    gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
