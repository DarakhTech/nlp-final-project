import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# --------------------------------------------------------------------------------------
# Constants (mirrors the notebook)
# --------------------------------------------------------------------------------------
ALL_CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

SUBSET_SIZES = [4, 6, 10, 12, 16, 20]
MODEL_CHOICES = ["Naive Bayes", "Logistic Regression", "Linear SVM"]
FEATURE_CHOICES = ["Bag of Words", "TF-IDF"]
VERSION_CHOICES = ["raw", "clean"]
RESULTS_CACHE_FILE = Path("precomputed_results.csv")


# --------------------------------------------------------------------------------------
# Preprocessing (same logic as the notebook)
# --------------------------------------------------------------------------------------
import nltk  # noqa: E402  (import after streamlit so UI starts quickly)
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, strip stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    clean_tokens: List[str] = []
    for word in text.split():
        if word not in STOP_WORDS:
            clean_tokens.append(LEMMATIZER.lemmatize(word))
    return " ".join(clean_tokens)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_datasets() -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Load raw + cleaned subsets for every k. Cached to avoid re-downloads."""
    raw_ds: Dict[int, Dict] = {}
    clean_ds: Dict[int, Dict] = {}

    for k in SUBSET_SIZES:
        cat_list = ALL_CATEGORIES[:k]
        train_raw = fetch_20newsgroups(
            subset="train", categories=cat_list, remove=("headers", "footers", "quotes")
        )
        test_raw = fetch_20newsgroups(
            subset="test", categories=cat_list, remove=("headers", "footers", "quotes")
        )

        X_train_raw, y_train_raw = train_raw.data, train_raw.target
        X_test_raw, y_test_raw = test_raw.data, test_raw.target

        raw_ds[k] = {
            "train_text": X_train_raw,
            "train_labels": y_train_raw,
            "test_text": X_test_raw,
            "test_labels": y_test_raw,
            "class_names": train_raw.target_names,
        }

        X_train_clean = [preprocess_text(t) for t in X_train_raw]
        X_test_clean = [preprocess_text(t) for t in X_test_raw]

        clean_ds[k] = {
            "train_text": X_train_clean,
            "train_labels": y_train_raw,
            "test_text": X_test_clean,
            "test_labels": y_test_raw,
            "class_names": train_raw.target_names,
        }

    return raw_ds, clean_ds


def get_dataset(k: int, version: str) -> Dict:
    raw_ds, clean_ds = load_datasets()
    return raw_ds[k] if version == "raw" else clean_ds[k]


# --------------------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------------------
def make_pipeline(model_choice: str, feature_choice: str) -> Pipeline:
    """Build a sklearn Pipeline for the requested model/feature combo."""
    if feature_choice == "Bag of Words":
        vectorizer = CountVectorizer()
    else:
        # 1–2 gram TF-IDF mirrors the notebook's strongest setup
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)

    if model_choice == "Naive Bayes":
        clf = MultinomialNB()
    elif model_choice == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = LinearSVC()

    return Pipeline([("vect", vectorizer), ("clf", clf)])


# --------------------------------------------------------------------------------------
# Training / evaluation with caching
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_and_eval(
    model_choice: str, feature_choice: str, k: int, version: str
) -> Dict:
    """Train once per combo and cache the fitted pipeline + metrics."""
    ds = get_dataset(k, version)
    X_train, y_train = ds["train_text"], ds["train_labels"]
    X_test, y_test = ds["test_text"], ds["test_labels"]

    pipe = make_pipeline(model_choice, feature_choice)
    start = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - start

    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    # Store per-class metrics for later tables/plots
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, preds, labels=range(len(ds["class_names"]))
    )
    per_class = pd.DataFrame(
        {
            "class": ds["class_names"],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }
    )

    return {
        "pipeline": pipe,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "train_time": train_time,
        "per_class": per_class,
        "y_true": y_test,
        "y_pred": preds,
        "class_names": ds["class_names"],
    }


@st.cache_resource(show_spinner=True)
def build_results_table() -> pd.DataFrame:
    """Recreate the experiment table across all k / versions / models / features."""
    rows = []
    for k in SUBSET_SIZES:
        for version in VERSION_CHOICES:
            for model_choice in MODEL_CHOICES:
                for feature_choice in FEATURE_CHOICES:
                    res = train_and_eval(model_choice, feature_choice, k, version)
                    rows.append(
                        {
                            "num_topics": k,
                            "version": version,
                            "model": model_choice,
                            "feature": feature_choice,
                            "accuracy": res["accuracy"],
                            "macro_f1": res["macro_f1"],
                            "train_time_sec": res["train_time"],
                        }
                    )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_cached_results() -> pd.DataFrame:
    """Load cached experiment table from disk, computing it once if missing."""
    if RESULTS_CACHE_FILE.exists():
        return pd.read_csv(RESULTS_CACHE_FILE)

    df = build_results_table()
    df.to_csv(RESULTS_CACHE_FILE, index=False)
    return df


def recompute_and_cache_results() -> pd.DataFrame:
    """Force a fresh computation and refresh the on-disk cache."""
    df = build_results_table()
    df.to_csv(RESULTS_CACHE_FILE, index=False)
    load_cached_results.clear()
    return df


# --------------------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------------------
def infer_text(
    text: str, model_choice: str, feature_choice: str, k: int, version: str
) -> Tuple[str, pd.DataFrame]:
    """Run inference with cached model and return label + score table."""
    res = train_and_eval(model_choice, feature_choice, k, version)
    pipe = res["pipeline"]
    class_names = res["class_names"]

    if version == "clean":
        text_to_use = preprocess_text(text)
    else:
        text_to_use = text

    # Predict label
    pred_idx = pipe.predict([text_to_use])[0]
    label = class_names[pred_idx]

    # Probability/confidence scores
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        probs = pipe.predict_proba([text_to_use])[0]
    else:
        # Use decision_function as a stand-in confidence and softmax-normalize
        scores = pipe.decision_function([text_to_use])[0]
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()

    score_df = pd.DataFrame({"class": class_names, "score": probs}).sort_values(
        "score", ascending=False
    )
    return label, score_df


# --------------------------------------------------------------------------------------
# Plotting utilities
# --------------------------------------------------------------------------------------
def plot_accuracy_lines(results: pd.DataFrame, version: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    subset = results[results["version"] == version].copy()
    subset["model_feature"] = subset["model"] + " / " + subset["feature"]
    sns.lineplot(
        data=subset,
        x="num_topics",
        y="accuracy",
        hue="model_feature",
        marker="o",
        ax=ax,
    )
    ax.set_title(f"Accuracy vs number of topics (version={version})")
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


def plot_confusion(res: Dict, class_names: List[str]):
    cm = confusion_matrix(res["y_true"], res["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names, rotation=0)
    plt.tight_layout()
    return fig


def plot_per_class_bars(per_class: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    melted = per_class.melt(
        id_vars=["class"], value_vars=["precision", "recall", "f1"], var_name="metric"
    )
    sns.barplot(data=melted, x="class", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Streamlit layout
# --------------------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="20 Newsgroups Topic Classifier",
        layout="wide",
        page_icon="📰",
    )

    st.title("20 Newsgroups Topic Classification – Classical NLP Showcase")
    st.markdown(
        """
This Streamlit app rebuilds the experiments from the accompanying notebook:

- Varies topic count k ∈ {4, 6, 10, 12, 16, 20}
- Compares preprocessing (raw vs. cleaned text)
- Tests classical models (Naive Bayes, Logistic Regression, Linear SVM)
- Explores feature choices (Bag of Words vs TF-IDF)

Use the tabs below to (1) try live inference and (2) browse the experimental results.
        """
    )

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            "All models and datasets are cached so you can explore combinations without retraining."
        )

    tabs = st.tabs(["Try the Model", "Experimental Results"])

    # ----------------------------------------------------------------------------------
    # Tab 1: Interactive inference
    # ----------------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Try the Model")
        st.markdown(
            "Paste text, pick a model/feature/k, and see the predicted topic with scores."
        )
        with st.popover("Sample Text Links"):
            st.header("Sci.space")
            st.markdown("https://www.nasa.gov/missions/")
            st.markdown("https://en.wikipedia.org/wiki/Space_exploration")

            st.header("Sci.med")
            st.markdown("https://www.who.int/news-room/fact-sheets")
            st.markdown("https://en.wikipedia.org/wiki/Medical_diagnosis")

            st.header("Rec.sport")
            st.markdown("https://www.fifa.com/tournaments/mens/worldcup")
            st.markdown("https://en.wikipedia.org/wiki/Association_football")

            st.header("Rec.autos")

            st.markdown("https://www.motortrend.com/cars/")
            st.markdown("https://en.wikipedia.org/wiki/Automobile_engine")


            st.header("Talk.politics")

            st.markdown("https://www.nytimes.com/international/section/politics")
            st.markdown("https://en.wikipedia.org/wiki/Political_ideology")
        

        text = st.text_area(
            "Input text",
            height=200,
            placeholder="Paste any news-style text here...",
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model_choice = st.selectbox("Model", MODEL_CHOICES, index=2)
        with col2:
            feature_choice = st.selectbox("Feature type", FEATURE_CHOICES, index=1)
        with col3:
            k_choice = st.selectbox("Number of classes (k)", SUBSET_SIZES, index=5)
        with col4:
            version_choice = st.selectbox("Text version", VERSION_CHOICES, index=1)

        if st.button("Run inference", type="primary"):
            if not text.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Running model..."):
                    label, score_df = infer_text(
                        text, model_choice, feature_choice, k_choice, version_choice
                    )
                st.success(f"Predicted topic: **{label}**")
                st.caption("Higher scores indicate higher confidence.")
                st.dataframe(score_df.reset_index(drop=True), use_container_width=True)

    # ----------------------------------------------------------------------------------
    # Tab 2: Experimental results
    # ----------------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Experimental Results Dashboard")
        st.markdown(
            "Plots and tables are recomputed from the same pipelines used in the notebook."
        )

        version_filter = st.selectbox(
            "Select data version for plots", VERSION_CHOICES, index=1, key="plot_version"
        )

        results_df = load_cached_results()
        st.caption(
            "Showing cached experimental metrics. Use 'Recalculate metrics' to refresh."
        )

        if st.button("Recalculate metrics", type="secondary"):
            with st.spinner("Recomputing all experiment metrics..."):
                results_df = recompute_and_cache_results()
            st.success("Metrics recomputed and cached for future loads.")

        st.markdown("### Accuracy vs number of topics (by model + feature)")
        fig_acc = plot_accuracy_lines(results_df, version_filter)
        st.pyplot(fig_acc, clear_figure=True)

        st.markdown("---")
        st.markdown("### Confusion matrix explorer")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cm_model = st.selectbox("Model", MODEL_CHOICES, index=2, key="cm_model")
        with c2:
            cm_feature = st.selectbox(
                "Feature type", FEATURE_CHOICES, index=1, key="cm_feature"
            )
        with c3:
            cm_k = st.selectbox("k", SUBSET_SIZES, index=2, key="cm_k")
        with c4:
            cm_version = st.selectbox(
                "Text version", VERSION_CHOICES, index=1, key="cm_version"
            )

        with st.spinner("Computing confusion matrix..."):
            res_cm = train_and_eval(cm_model, cm_feature, cm_k, cm_version)
        fig_cm = plot_confusion(res_cm, res_cm["class_names"])
        st.pyplot(fig_cm, clear_figure=True)

        st.markdown("---")
        st.markdown("### Per-class Precision / Recall / F1")
        st.caption("Values are computed on the selected (model, feature, k, version).")
        with st.spinner("Computing per-class metrics..."):
            res_pc = train_and_eval(cm_model, cm_feature, cm_k, cm_version)
            per_class_df = res_pc["per_class"]
        st.dataframe(per_class_df, use_container_width=True)
        fig_pc = plot_per_class_bars(per_class_df)
        st.pyplot(fig_pc, clear_figure=True)

    st.markdown("---")
    st.markdown(
        """
**Notes & limitations**
- Classical models with bag-of-words assumptions; no contextual embeddings.
- TF-IDF + Linear SVM is strongest here, but probabilities are approximate for SVM.
- Results may differ slightly run-to-run due to training randomness and caching.
        """
    )


if __name__ == "__main__":
    main()

