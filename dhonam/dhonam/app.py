"""Streamlit interface for the Dholuo agricultural topic classifier.

Run locally with:  streamlit run app.py
The model is loaded from models/final/ (or the directory in DHOLUO_MODEL_DIR).
"""

import os

import pandas as pd
import streamlit as st

from src.prediction import InputValidationError, ModelArtifactError, TopicClassifier

st.set_page_config(page_title="Dholuo Agricultural Topic Classifier", layout="centered")

# Verified evaluation figures, taken from the executed notebook
# (notebooks/dholuo_topic_classification_final.ipynb, sections 13 and 14.6).
TEST_MACRO_F1 = 0.396
TEST_MACRO_F1_CI = "0.329 to 0.442"
UNSEEN_PROMPT_MACRO_F1 = 0.361
TEST_ACCURACY = 0.424
N_TEST_SPEAKERS = 5


@st.cache_resource(show_spinner="Loading model...")
def load_classifier(model_dir: str):
    return TopicClassifier(model_dir)


def sidebar():
    with st.sidebar:
        st.header("About this model")
        st.markdown(
            "A TF-IDF + logistic-regression classifier trained on 810 transcribed answers from 13 "
            "speakers in the Utavu Foundation Dholuo Speech Dataset. It was selected on validation "
            "data over a BiLSTM and fine-tuned AfroXLMR models, which did not score significantly higher."
        )
        st.markdown(
            f"**Tested on {N_TEST_SPEAKERS} speakers it never saw during training:**\n\n"
            f"- macro-F1 {TEST_MACRO_F1} (95% interval {TEST_MACRO_F1_CI})\n"
            f"- accuracy {TEST_ACCURACY}\n"
            f"- macro-F1 {UNSEEN_PROMPT_MACRO_F1} on answers to questions not seen in training, "
            "the more realistic figure for new questions"
        )
        st.header("Limitations")
        st.markdown(
            "- Agriculture only, 23 topics. Any input gets one of these topics, even if it is off-topic.\n"
            "- Labels come from the question each speaker answered, not from annotating the answer.\n"
            "- Trained on 13 speakers, mostly aged 18 to 29. It may work less well for other speakers or regions.\n"
            "- Roughly 4 in 10 answers are classified correctly. Treat the output as a suggestion."
        )
        st.header("Responsible use")
        st.markdown(
            "Use as a routing or search aid with a person checking the result. Do not use it to make "
            "decisions about people, such as eligibility for support or subsidies."
        )
        st.caption(
            "Data: Utavu Foundation and CLEAR Global, TWB Voice platform (NOODL-1.0 licence). "
            "Text you enter is processed in memory and not stored by this app."
        )


def show_prediction(clf: TopicClassifier, pred) -> None:
    st.subheader("Most likely topic")
    st.markdown(f"### {pred.label}")
    st.caption(
        "This is the agricultural sub-domain the model associates with the text, using the dataset's "
        "question-based topic labels. It is not an independent judgement of what the answer is about."
    )

    for w in pred.warnings:
        st.warning(w)

    st.subheader("Top three topics")
    top = pred.top(3)
    if pred.calibrated:
        df = pd.DataFrame({"Topic": [t for t, _ in top], "Confidence": [s for _, s in top]})
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"Confidence": st.column_config.ProgressColumn(
                         "Confidence", format="%.0f%%", min_value=0.0, max_value=1.0)})
        st.caption("Confidences are calibrated on the validation speakers with temperature scaling.")
    else:
        for rank, (topic, _) in enumerate(top, start=1):
            st.markdown(f"{rank}. {topic}")
        st.info(
            "Confidence percentages are not shown. On the test speakers this model's raw probabilities "
            "were badly under-confident (average 11% for a 42% hit rate), so they would mislead. They will "
            "appear here once a calibration temperature fitted on validation data is added to the model "
            "files (see models/README.md)."
        )
        with st.expander("Raw model scores (uncalibrated, for inspection only)"):
            st.dataframe(pd.DataFrame(pred.ranking, columns=["Topic", "Raw score"]),
                         hide_index=True, width="stretch")

    with st.expander("How the text was processed"):
        st.markdown(
            f"The model was trained on the `{clf.text_variant}` text variant: Unicode normalisation, straight "
            "apostrophes, lower case, consistent punctuation spacing, and one token each for code-switching "
            "(`<cs>`), pauses (`<pause>`) and unclear speech (`<unclear>`)."
        )
        st.code(pred.normalized_text, language=None)
        cov = "not available" if pred.vocab_coverage is None else f"{pred.vocab_coverage:.0%}"
        st.markdown(f"Words: **{pred.n_words}**. Share of words known to the model: **{cov}**.")


def main():
    sidebar()
    st.title("Dholuo Agricultural Topic Classifier")
    st.markdown(
        "Paste or type a Dholuo answer about farming, for example a transcribed reply to an extension "
        "officer's question. The model suggests which of 23 agricultural topics it belongs to, such as "
        "poultry farming, fish farming or water use and irrigation."
    )
    with st.expander("What it does and does not do"):
        st.markdown(
            "**Does:** assign Dholuo text about agriculture to one of 23 sub-domains used in the dataset, "
            "and show the next most likely topics.\n\n"
            "**Does not:** transcribe audio, translate, give farming advice, detect whether the text is "
            "Dholuo, or say that a text is outside agriculture. Transcripts may use the dataset's markers "
            "such as `[cs]...[cs]` for code-switched words, `[pause]` and `[?]`, but plain text works too."
        )

    model_dir = os.environ.get("DHOLUO_MODEL_DIR", "models/final")
    try:
        clf = load_classifier(model_dir)
    except ModelArtifactError as exc:
        st.error(f"The model could not be loaded. {exc}")
        st.stop()

    with st.form("classify"):
        text = st.text_area("Dholuo transcript", height=180, max_chars=10_000,
                            placeholder="Type or paste a Dholuo transcript here")
        submitted = st.form_submit_button("Classify", type="primary")

    if submitted:
        try:
            pred = clf.predict(text)
        except InputValidationError as exc:
            st.warning(str(exc))
        except Exception as exc:  # unexpected inference failure: report, do not crash the page
            st.error(f"Something went wrong during prediction ({type(exc).__name__}). Please try again.")
        else:
            show_prediction(clf, pred)

    st.divider()
    st.caption(f"Model run: {clf.info()['run_id']} | Text variant: {clf.text_variant} | "
               f"Calibrated confidences: {'yes' if clf.calibrated else 'no'}")


main()
