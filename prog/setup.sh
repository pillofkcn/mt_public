#!/usr/bin/env bash
# =============================================================================
# Setup of the syntactic analysis project, version 1.0.
#
#   1. Checks the Python version (3.10-3.13; 3.14 breaks spaCy via pydantic v1)
#   2. Creates the virtual environment .venv and installs the project: with
#      LOCKED=1 at the exact versions of version 1.0 (requirements-lock.txt),
#      otherwise at the newest versions the ranges of pyproject.toml allow
#   3. Downloads every published model a run loads and checks that it loads
#   4. Downloads UD_English-EWT (release r2.17) and checks every file against
#      data/en_ewt-ud-*.conllu.sha256
#   5. Downloads the confirmatory test sets GENTLE and MCTB_en
#      (scripts/fetch_data.py) and checks them against data/*.sha256
#   6. Trains the five models of plan/v1.yaml that this project trains; about
#      3 h 15 min on a CPU when none is there yet, seconds when they all are
#
# Usage, from prog/ (on Windows in Git Bash):
#   bash setup.sh
#   LOCKED=1 bash setup.sh             # the exact versions of version 1.0 (Python 3.13)
#   PYTHON=python3.13 bash setup.sh    # another interpreter
#   PYTHON="py -3.13" bash setup.sh    # the Windows launcher
# =============================================================================
# -e  exit on error   -u  error on unset variable   -o pipefail  catch failures
# mid-pipeline: without it a failing download piped into anything would be
# reported as success.
set -euo pipefail

VENV_DIR=".venv"

# The interpreter that creates the environment. PYTHON may be a command with
# arguments, such as the Windows launcher "py -3.13", so it is split into words.
if [ -n "${PYTHON:-}" ]; then
    read -r -a PY <<< "$PYTHON"
elif python3 -c "" &>/dev/null; then
    PY=(python3)
else
    PY=(python)
fi

# UD_English-EWT release r2.17. Its train, dev and test files are byte for byte
# the ones every number of version 1.0 was measured on (data/*.sha256,
# data/DATASET.md). Never "master" or "dev": the corpus changes between
# releases (relation renames, re-attachments), and every score with it.
EWT_TAG="${EWT_TAG:-r2.17}"

# 1 installs requirements-lock.txt, the exact versions of the environment of
# version 1.0 (Python 3.13.0, Windows, CPU only), instead of resolving the
# ranges of pyproject.toml: the way to reproduce version 1.0.
LOCKED="${LOCKED:-0}"

PY_VERSION=$("${PY[@]}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("${PY[@]}" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("${PY[@]}" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -lt 10 ] || [ "$PY_MINOR" -gt 13 ]; then
    echo "Error: Python 3.10-3.13 is required (found $PY_VERSION)"
    echo "       Python 3.14 is not supported: spaCy still requires pydantic v1."
    exit 1
fi
echo "Using Python $PY_VERSION (${PY[*]})"
# The lock was taken on Python 3.13.0, and several of its versions (numpy 2.4,
# scipy 1.17, pandas 3.0) do not install on 3.10.
if [ "$LOCKED" = "1" ] && [ "$PY_MINOR" -ne 13 ]; then
    echo "Error: LOCKED=1 installs requirements-lock.txt, taken on Python 3.13;"
    echo "       found $PY_VERSION. Use PYTHON=python3.13, or PYTHON=\"py -3.13\" on Windows."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "${PY[@]}" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate (Unix first, then Windows/Git Bash)
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"

echo "Installing dependencies..."
if [ "$LOCKED" = "1" ]; then
    # Every package at the version of the environment of version 1.0, pip
    # itself among them, then this project without resolving anything again.
    python -m pip install -r requirements-lock.txt
    python -m pip install -e . --no-deps
else
    python -m pip install --upgrade pip
    # .[all]: every analyzer a stage runs, and matplotlib for the figures. The
    # versions are the newest the ranges of pyproject.toml allow, not
    # necessarily those of version 1.0 (requirements-lock.txt, LOCKED=1).
    python -m pip install -e ".[all]"
fi

mkdir -p data model_cache

# --- Published models ---------------------------------------------------------
# Every download is followed by a check that the model is there. A download that
# fails silently would surface only when a run loads the model, hours later.

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm
python -c "import spacy; spacy.load('en_core_web_sm')" \
    || { echo "Error: en_core_web_sm did not load after download."; exit 1; }

echo "Downloading NLTK data..."
python - <<'PY'
import nltk
for pkg in ("punkt_tab", "averaged_perceptron_tagger_eng", "treebank"):
    nltk.download(pkg, quiet=True)
# Fail loudly rather than let a missing corpus surface later as a 0.0 score.
for path in ("tokenizers/punkt_tab",
             "taggers/averaged_perceptron_tagger_eng",
             "corpora/treebank"):
    nltk.data.find(path)
print("NLTK data OK.")
PY

echo "Downloading Benepar model..."
python - <<'PY'
import nltk, sys
try:
    nltk.data.find("models/benepar_en3")
except LookupError:
    try:
        import benepar
        benepar.download("benepar_en3")
    except AttributeError:
        # Known sha256 bug in NLTK's downloader. Swallowing it here is only safe
        # if the model really is on disk, which the check below enforces.
        pass
try:
    nltk.data.find("models/benepar_en3")
except LookupError:
    sys.exit("Error: benepar_en3 is missing and could not be downloaded.")
print("Benepar model OK.")
PY

echo "Downloading Stanza models and ELECTRA-large; loading Benepar..."
python - <<'PY'
import gc

from syntactic_analysis.build import build
from syntactic_analysis.plan import load

# Every configuration of plan/v1.yaml that runs a Stanza or a Benepar model,
# built and loaded by build.build, the path a run takes, so this downloads
# what a run loads, where it looks for it, and nothing else:
#   * Stanza: load_model() runs stanza.download() for the processors and the
#     models of the row - packages ewt and combined, pos and depparse
#     combined_electra-large (stanza_trf), constituency ptb3-revised_charlm
#     and ptb3-revised_electra-large - into Stanza's resources directory
#     (STANZA_RESOURCES_DIR, or Stanza's user cache), and refuses a pipeline
#     that loaded another model than the row declares.
#   * ELECTRA-large: the electra-large models read the transformer
#     google/electra-large-discriminator, which is not one of Stanza's
#     downloads. The wrapper loads it through Stanza's own loader, from the
#     Hugging Face cache (HF_HOME, or ~/.cache/huggingface), and fetches it
#     there when it is missing (about 1.3 GB). It refuses any revision other
#     than the one wrappers/stanza.py records (_TRANSFORMER_REVISIONS): if the
#     hub has moved on from it, setup stops here rather than accept another.
#   * Benepar: benepar_en3, downloaded above, over en_core_web_sm.
# The loaded pipeline is the check that each model landed.
plan = load(strict=True)
for key, configuration in plan.configurations.items():
    if configuration.wrapper.split(".")[0] not in ("stanza", "benepar"):
        continue
    print(f"  {key} ...", flush=True)
    analyzer = build(plan, key)
    print(f"    {analyzer.model_version}", flush=True)
    del analyzer
    gc.collect()  # ELECTRA-large is 1.3 GB; one row in memory at a time
print("Stanza models, ELECTRA-large and Benepar OK.")
PY

echo "Downloading SuPar model..."
python - <<'PY'
import torch, warnings
warnings.filterwarnings("ignore")
# SuPar 1.1.4 predates torch 2.6's weights_only=True default, so its checkpoints
# cannot be loaded without this shim. It unpickles code from a remote file, so
# it is scoped to this call only and restored immediately after.
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
try:
    from supar import Parser
    # The checkpoint of supar_con: PTB constituency (CharLSTM).
    for name in ("crf-con-en",):
        print(f"  {name} ...", flush=True)
        Parser.load(name)
finally:
    torch.load = _orig
print("SuPar model OK.")
PY

# UDPipe and RDRPOSTagger: nothing to download. The models udpipe_ewt and
# rdr_ewt run are trained on the EWT train split below.

# --- UD_English-EWT -----------------------------------------------------------
# curl needs -f: without it an HTTP 404/500 body is written into the .conllu file
# and curl still exits 0, so `set -e` cannot catch it, and an HTML error page
# would parse as zero sentences.
EWT_BASE="https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/${EWT_TAG}"
for split in train dev test; do
    target="data/en_ewt-ud-${split}.conllu"
    if [ ! -f "$target" ]; then
        echo "Downloading $target (${EWT_TAG})..."
        curl -fSL --retry 3 --retry-delay 2 \
             "$EWT_BASE/en_ewt-ud-${split}.conllu" -o "${target}.part"
        mv "${target}.part" "$target"
    fi
    # The file must match its recorded digest: a changed corpus changes every score.
    (cd data && sha256sum -c "en_ewt-ud-${split}.conllu.sha256" >/dev/null) \
        || { echo "Error: $target does not match data/en_ewt-ud-${split}.conllu.sha256."
             echo "       The recorded digests are those of release r2.17, which every"
             echo "       number of version 1.0 was measured on. Delete the file and run"
             echo "       setup.sh again; with another EWT_TAG this check fails by design"
             echo "       (data/DATASET.md)."
             exit 1; }
done

# A truncated or wrong-format corpus must not reach the experiments: the test
# split must load with exactly the counts data/DATASET.md records.
python - <<'PY'
from syntactic_analysis.data_loader import load_conllu
from syntactic_analysis.evaluation.gold import EWT_TEST_SENTENCES, EWT_TEST_TOKENS
gold = load_conllu("data/en_ewt-ud-test.conllu")
n_sent = len(gold.sentences)
n_tok = sum(len(s.tokens) for s in gold.sentences)
if (n_sent, n_tok) != (EWT_TEST_SENTENCES, EWT_TEST_TOKENS):
    raise SystemExit(
        f"Error: test split loaded {n_sent} sentences / {n_tok} tokens, but "
        f"data/DATASET.md records {EWT_TEST_SENTENCES} / {EWT_TEST_TOKENS}."
    )
print(f"Gold standard OK: {n_sent} sentences / {n_tok} tokens.")
PY

# --- Confirmatory test sets ---------------------------------------------------
# GENTLE (UD 2.18) and MCTB_en, from the sources plan/v1.yaml pins, each file
# checked against data/*.sha256; a file that does not match is refused and left
# as it is. The script prints counts and digests only: the sets are sealed, and a
# stage opens them only in a confirmatory run (README.md).
echo "Downloading the confirmatory test sets (GENTLE, MCTB_en)..."
python scripts/fetch_data.py

# --- Models trained in this project -------------------------------------------
# Five configurations of plan/v1.yaml run a model trained here on the EWT train
# split; none of the scripts reads the test split. A build or a run never trains
# a model: a missing one fails its configuration and names its training script.
# Each script does nothing when its model is in model_cache/ already (--force
# trains again). Times on the CPU of the original runs (Intel64 Family 6
# Model 183); they depend on the load of the machine:
#   udpipe_ewt          scripts/train_udpipe.py  about 100 min
#   nltk_ewt, nltk_arc  scripts/train_nltk.py    about 25 s and 6 min
#   spacy_ewt           scripts/train_spacy.py   about 70 min
#   rdr_ewt             scripts/train_rdrpos.py  about 20 min; it downloads the
#                       authors' learner (GPL) at a pinned commit into
#                       ~/.cache/rdrpostagger, outside the project
echo "Training the models of plan/v1.yaml trained in this project"
echo "(about 3 h 15 min on a CPU when none is there yet)..."
python scripts/train_udpipe.py
python scripts/train_nltk.py
python scripts/train_spacy.py
python scripts/train_rdrpos.py
python - <<'PY'
from syntactic_analysis.build import trained_models
from syntactic_analysis.plan import load

# Every model the plan needs from this project is on disk now: each
# configuration whose wrapper declares a trained model has all its files.
missing = {
    key: [str(path) for path in model.missing()]
    for key, model in trained_models(load(strict=True)).items()
    if model.missing()
}
if missing:
    raise SystemExit(f"Error: models still missing after training: {missing}")
print("Models trained in this project OK.")
PY

echo ""
echo "========================================="
echo " Setup complete!"
echo "========================================="
echo ""
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate        (Linux/Mac)"
echo "  source $VENV_DIR/Scripts/activate    (Windows, Git Bash)"
echo "Then, from prog/:"
echo "  python scripts/smoke_plan.py         # every configuration on 20 sentences"
echo "  python scripts/run_pipeline.py       # the exploratory run (about 2 h)"
