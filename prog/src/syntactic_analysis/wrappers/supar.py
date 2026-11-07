"""SuPar (Suzhou Parser) — constituency parsing.

SuPar is a syntactic parser developed at Soochow University. It combines
Conditional Random Fields (CRF) with neural encoders. Published at ACL 2020 and
IJCAI 2020.

How it works:
  1. Input text is encoded by a CharLSTM encoder
  2. The encoder output is fed into a CRF constituency decoder, which produces
     the tree

Key characteristics:
  - This wrapper runs SuPar's constituency model only: `crf-con-en`, PTB
    constituency with a CharLSTM encoder, the supar_con row of plan/v1.yaml
  - Per-model metadata (scheme, encoder, training corpus) lives in the
    _CON_MODELS registry below
  - Models auto-download on first load_model() call, from the GitHub releases
    of SuPar into SuPar's cache (~/.cache/supar), except during a run, which
    downloads nothing (offline.py). Until 24 Oct 2026 this line said the
    Hugging Face hub; SuPar's own table (supar.MODEL["github"]) says GitHub.
  - Natively accepts both raw text and pre-tokenized input (list[list[str]])
  - Output follows CoNLL-U column layout (ID, FORM, LEMMA, UPOS, ...)

Until 23 Oct 2026 the wrapper also ran SuPar's dependency models
(`biaffine-dep-en`, PTB Stanford Dependencies, with the SD->UD label mapping;
`biaffine-dep-xlmr`, UD v2.3) and declared a second constituency model,
`crf-con-roberta-en`, which no released supar can load. Version 1.0 runs SuPar
for constituency only, so they were removed. `dep_model` stays a parameter,
because plan/v1.yaml states `dep_model: null` for supar_con, and accepts None
alone.

Install: pip install supar
"""

from __future__ import annotations

from pathlib import Path

from syntactic_analysis.base import (
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    MethodLevel,
    MethodType,
    TrainingDomain,
)
from syntactic_analysis.normalize import (
    collapse_to_single_line,
    escape_bracket_leaves_tree,
)
from syntactic_analysis.models import (
    ConstituencyParse,
    DependencyParse,
)
from syntactic_analysis.downloads import DownloadedFiles, supar_cache
from syntactic_analysis.offline import downloads_allowed, not_cached

# Per-model metadata, keyed by the exact string passed to supar.Parser.load().
# This is the single place that knows how a SuPar model maps onto the
# framework's categorisation axes; supported_types / annotation_scheme /
# method_detail / training_corpus are all *derived* from the entry the instance
# was constructed with.
_CON_MODELS: dict[str, dict] = {
    "crf-con-en": {
        "scheme": AnnotationScheme.PTB,
        "method_detail": "CRF (CharLSTM encoder)",
        "training_corpus": "PTB WSJ 02-21 (Penn Treebank constituency)",
        "level": MethodLevel.NEURAL_RECURRENT,
        "domain": TrainingDomain.OUT,
    },
}

# The one value dep_model accepts: no dependency model.
_DEP_MODELS: tuple[None, ...] = (None,)


class SuparAnalyzer(BaseAnalyzer):

    def __init__(
        self,
        *,
        key: str | None = None,
        dep_model: None = None,
        con_model: str = "crf-con-en",
    ) -> None:
        """Create the analyzer.

        Args:
            key:       Key for this instance (becomes `.name`); "supar"
                       when None. plan/v1.yaml names the row supar_con.
            dep_model: None, the only value: SuPar's dependency models went with
                       version 1.0 (module docstring). A parameter still,
                       because the plan states it.
            con_model: Constituency model id, a key of _CON_MODELS.

        Raises:
            ValueError: If dep_model is not None, or con_model names a model not
                        present in _CON_MODELS.
        """
        if dep_model not in _DEP_MODELS:
            raise ValueError(
                f"unknown dep_model {dep_model!r}; expected one of {list(_DEP_MODELS)}"
            )
        if con_model not in _CON_MODELS:
            raise ValueError(
                f"unknown con_model {con_model!r}; expected one of {sorted(_CON_MODELS)}"
            )
        self._key = key or "supar"
        self._con_model_name = con_model
        self._con_parser = None  # set by load_model(); also doubles as the "is it loaded" flag

    @property
    def name(self) -> str:
        return self._key

    @property
    def supported_types(self) -> set[AnalysisType]:
        return {AnalysisType.CONSTITUENCY_PARSING}

    @property
    def method_type(self) -> MethodType:
        return MethodType.NEURAL

    @property
    def method_detail(self) -> str:
        return f"con: {_CON_MODELS[self._con_model_name]['method_detail']}"

    @property
    def display_name(self) -> str:
        return "SuPar"

    @property
    def annotation_scheme(self) -> dict:
        return {
            AnalysisType.CONSTITUENCY_PARSING: _CON_MODELS[self._con_model_name]["scheme"]
        }

    @property
    def method_level(self) -> dict:
        # The CharLSTM-encoder model is BiLSTM-based (L3).
        return {
            AnalysisType.CONSTITUENCY_PARSING: _CON_MODELS[self._con_model_name]["level"]
        }

    @property
    def training_domain(self) -> dict:
        return {
            AnalysisType.CONSTITUENCY_PARSING: _CON_MODELS[self._con_model_name]["domain"]
        }

    @property
    def training_corpus(self) -> str:
        info = _CON_MODELS[self._con_model_name]
        return f"{info['training_corpus']} ({self._con_model_name})"

    @property
    def model_version(self) -> str:
        """Concrete loaded-model identity, for provenance logging.

        SuPar's registry name doubles as the version identifier — there is no
        separate semantic version stamped on the downloaded checkpoint beyond
        the string used to fetch it (e.g. "crf-con-en").

        Returns:
            Empty string if load_model() has not been called yet.
        """
        if self._con_parser is None:
            return ""
        return f"con={self._con_model_name}"

    def downloaded_files(self, model_name: str = "", **kwargs) -> tuple[DownloadedFiles, ...]:
        """The files of SuPar's cache load_model reads.

        Takes load_model's keywords. Found as Parser.load finds them
        (supar/parsers/parser.py, Parser.load, and supar.utils.fn.download):
        the release's zip file, named by the last part of the URL supar.MODEL
        gives the model on GitHub, which download() opens to read the name of
        the one file it holds, and that file, extracted beside it, which
        torch.load reads. When the zip is not there only the zip is declared,
        and the lock check reports it missing. Under supar_cache. Nothing is
        loaded. Since 24 Oct 2026.
        """
        import os
        import urllib.parse
        import zipfile

        import supar

        cache = supar_cache()
        url = supar.MODEL["github"].get(self._con_model_name, self._con_model_name)
        archive = cache / os.path.basename(urllib.parse.urlparse(url).path)
        files = [archive]
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                files.append(cache / handle.infolist()[0].filename)
        return (DownloadedFiles.under("supar_cache", cache, files),)

    def load_model(self, model_name: str = "", **kwargs) -> None:
        """Load the constituency model.

        Args:
            model_name: Refused when not empty: the model is named in the
                       constructor (con_model).

        SuPar loads the model from its cache (~/.cache/supar) and downloads it
        there first when it is missing (supar.utils.fn.download, through
        torch.hub.download_url_to_file). During a run (offline.py) that
        download is refused for the duration of the load: a missing model
        fails the configuration instead of being fetched.

        Raises:
            TypeError:  If model_name is not empty, or an unknown keyword
                        argument is passed.
            offline.NotCachedError: During a run, if the model is not in
                        SuPar's cache.
        """
        self._reject_unknown_options(kwargs)
        if model_name:
            raise TypeError(
                f"SuparAnalyzer takes its model from the constructor (con_model=), "
                f"not from model_name (got {model_name!r})"
            )

        import torch  # needed to patch weights_only default for PyTorch 2.6+
        import torch.hub
        # deferred import so supar is only required at runtime
        from supar import Parser

        name = self._con_model_name

        def refuse_download(url, dst, *args, **kwargs):
            raise not_cached(f"the SuPar model {name!r} ({Path(dst).name})",
                             "SuPar's cache (~/.cache/supar)")

        # PyTorch 2.6+ defaults to weights_only=True which breaks SuPar's checkpoint loading
        _original_load = torch.load
        _original_fetch = torch.hub.download_url_to_file
        torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})
        if not downloads_allowed():
            torch.hub.download_url_to_file = refuse_download
        try:
            self._con_parser = Parser.load(name)
        finally:
            torch.load = _original_load  # restore original torch.load
            torch.hub.download_url_to_file = _original_fetch

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Not supported — this wrapper runs SuPar's constituency model only.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "SuparAnalyzer parses constituency only; SuPar's dependency models "
            "are not part of version 1.0."
        )

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Run the CRF constituency parser.

        Args:
            text: Either a raw string or a list of pre-tokenized sentences.
                  SuPar's predict() method accepts both formats natively.

        Returns a ConstituencyParse with one single-line bracketed tree per sentence.
        The tree is stored in column [2] of SuPar's CoNLL-U-style output.

        Bracket characters in the input are escaped to their PTB forms BEFORE the
        tree is serialised. SuPar prints a literal "(" token verbatim, and at that
        point it is indistinguishable from a structural bracket: `Tree.fromstring`
        does not raise, it silently drops leaves and shifts every span offset. The
        gold test set contains 196 such tokens across 80 sentences (3.85%), and
        because the corrupted spans then fed the inter-parser consensus, one
        wrapper defect degraded all four parsers' scores at once.

        Raises:
            RuntimeError: If the constituency model was not loaded.
        """
        self._ensure_loaded()
        # lang=None for pretokenized input; lang="en" would re-tokenize and break word lists
        lang = None if isinstance(text, list) else "en"
        dataset = self._con_parser.predict(text, lang=lang, verbose=False)
        trees = []
        for sent_result in dataset.sentences:
            tree = sent_result.values[2]  # column [2] = Tree (not [-1], the chart matrix)
            # Escape on the Tree object, while the structure is still known.
            tree = escape_bracket_leaves_tree(tree)
            # str(Tree) pretty-prints across several lines; the data model promises
            # one tree per line.
            trees.append(collapse_to_single_line(str(tree)))
        return ConstituencyParse(trees=trees, analyzer_name=self.name)

    def _ensure_loaded(self) -> None:
        """Guard analysis calls behind a loaded model."""
        self._require_loaded(self._con_parser is not None)
