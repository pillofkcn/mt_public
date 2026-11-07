"""Benepar (Berkeley Neural Parser) — constituency parsing on top of spaCy.

Benepar is a chart-based neural constituency parser that integrates into
spaCy as a pipeline component. It adds constituency parsing capability
to spaCy, which natively only supports dependency parsing and POS tagging.

How it works:
  1. A standard spaCy pipeline handles tokenization, POS tagging, and
     dependency parsing — the pipeline SpacyAnalyzer runs, so tags and arcs
     agree wherever the two build the same Doc. They do not for pre-tokenised
     input that spaCy would re-segment (65 of the 2077 sentences of the EWT
     test split): this wrapper pins sentence starts and SpacyAnalyzer does not
     (see `_pretokenized_doc`, and `_analyze_pretokenized` in wrappers/spacy.py)
  2. Benepar is added as an additional pipeline component ("benepar")
  3. For each sentence, Benepar runs a neural chart parser that produces
     a constituency tree accessible via the spaCy span extension: sent._.parse_string

Key characteristics:
  - Supports ALL three analysis types: POS, dependency (via spaCy), constituency
  - POS and dependency come from the underlying spaCy model. They match
    SpacyAnalyzer only when both are given the same map_tier; this wrapper
    previously emitted raw Stanford Dependencies while SpacyAnalyzer mapped
    them to UD, so the two silently disagreed on identical parses.
  - Constituency parsing is Benepar's own contribution
  - Requires a spaCy model to be pre-installed: python -m spacy download en_core_web_sm
  - Benepar model (~100 MB) auto-downloads on first load_model() call, except
    during a run, which downloads nothing (offline.py)

Install: pip install benepar
"""

from __future__ import annotations

from syntactic_analysis.base import (
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    MethodLevel,
    MethodType,
    TrainingDomain,
)
from syntactic_analysis.deprel_map import HEADLINE_TIER, MapTier, validate_tier
from syntactic_analysis.downloads import DownloadedFiles, nltk_data_directory
from syntactic_analysis.models import (
    ConstituencyParse,
    DependencyParse,
    Sentence,
)
from syntactic_analysis.normalize import (
    collapse_to_single_line,
    escape_bracket_leaves_tree,
)
from syntactic_analysis.offline import downloads_allowed, not_cached
from syntactic_analysis.spacy_doc import (
    PIPELINE_CORPUS,
    doc_to_depparse,
    doc_tokens,
    library_files,
    package_files,
    pipeline_level,
    pipeline_name,
)


# What this wrapper's models were trained on, per task. Only the constituency
# parser is Benepar's; the tags and the arcs are the spaCy pipeline's, and are
# described as the spaCy wrapper describes them (spacy_doc.PIPELINE_CORPUS).
# Until 16 Oct 2026 this module said "OntoNotes 5 via en_core_web_sm for POS
# and dependencies", so analyzers.csv gave one pipeline two corpus strings, and
# named a model load_model() lets the caller replace.
_CORPUS_CON = "PTB revised WSJ 02-21, T5-small encoder (benepar_en3)"
_CORPUS_POS_DEP = PIPELINE_CORPUS


class BeneparAnalyzer(BaseAnalyzer):

    def __init__(self, *, map_tier: MapTier = HEADLINE_TIER) -> None:
        """Create the analyzer.

        Args:
            map_tier: "none" | "casing" | "full" — SD->UD conversion for the
                dependency output, which comes from the underlying spaCy model.
                Matches SpacyAnalyzer's map_tier, so the two agree on identical
                parses. map_tier is the only switch, here as in SpacyAnalyzer,
                whose older map_to_ud was removed on 24 Oct 2026.
        """
        self._map_tier = validate_tier(map_tier)
        self._nlp = None  # spaCy pipeline with Benepar component attached
        self._benepar_model = "benepar_en3"  # overwritten by load_model()
        # Constituency input convention; see set_ptb_escaped_input().
        self._ptb_escaped_input = False
        self._treebank_parser = None  # benepar.Parser, built on first use

    @property
    def name(self) -> str:
        return "benepar"

    @property
    def supported_types(self) -> set[AnalysisType]:
        # all three analysis types: POS, dependency (spaCy), constituency (Benepar)
        return {
            AnalysisType.POS_TAGGING,
            AnalysisType.DEPENDENCY_PARSING,
            AnalysisType.CONSTITUENCY_PARSING,
        }

    @property
    def method_type(self) -> MethodType:
        return MethodType.NEURAL

    @property
    def method_detail(self) -> str:
        return "Chart-based neural"

    @property
    def display_name(self) -> str:
        return "Benepar"

    @property
    def annotation_scheme(self) -> dict:
        # POS and dependencies come from the underlying spaCy model, so they
        # inherit spaCy's schemes; only constituency is Benepar's own.
        return {
            AnalysisType.POS_TAGGING: AnnotationScheme.UD,
            AnalysisType.DEPENDENCY_PARSING: AnnotationScheme.SD,
            AnalysisType.CONSTITUENCY_PARSING: AnnotationScheme.PTB,
        }

    @property
    def method_level(self) -> dict:
        # benepar_en3 is a chart parser over a fine-tuned T5-small encoder — a
        # pretrained transformer, L4. POS and dependencies are the spaCy
        # pipeline's, at the level the spaCy wrapper gives the same pipeline
        # (spacy_doc.pipeline_level): L2 for en_core_web_sm, which is why Benepar
        # is only ever scored on constituency. Until 16 Oct 2026 they were L2
        # whichever pipeline was loaded.
        pipeline = pipeline_level(self._nlp)
        return {
            AnalysisType.POS_TAGGING: pipeline,
            AnalysisType.DEPENDENCY_PARSING: pipeline,
            AnalysisType.CONSTITUENCY_PARSING: MethodLevel.NEURAL_PRETRAINED,
        }

    @property
    def training_domain(self) -> dict:
        return {
            AnalysisType.POS_TAGGING: TrainingDomain.OUT,
            AnalysisType.DEPENDENCY_PARSING: TrainingDomain.OUT,
            AnalysisType.CONSTITUENCY_PARSING: TrainingDomain.OUT,
        }

    @property
    def training_corpus(self) -> str:
        return f"{_CORPUS_CON} / {_CORPUS_POS_DEP}"

    def training_corpus_for(self, analysis_type: AnalysisType) -> str:
        """The corpus of one task: Benepar's own for constituency, the spaCy
        pipeline's for POS and dependencies.

        The tool-wide `training_corpus` names both, so a per-task cell built
        from it would carry the other task's corpus as well.
        """
        if analysis_type is AnalysisType.CONSTITUENCY_PARSING:
            return _CORPUS_CON
        return _CORPUS_POS_DEP

    @property
    def model_version(self) -> str:
        """The Benepar model and the spaCy pipeline it is attached to.

        The pipeline is named as spaCy's own meta.json names it, language
        included, by the function the spaCy wrapper names it with
        (spacy_doc.pipeline_name). Until 16 Oct 2026 this wrapper had no
        version, so a run that loaded another `benepar_model` recorded the same
        corpus string as one that did not. Empty until load_model() has run.
        """
        if self._nlp is None:
            return ""
        return f"{self._benepar_model} on {pipeline_name(self._nlp)}"

    def downloaded_files(
        self, model_name: str = "en_core_web_sm", benepar_model: str = "benepar_en3", **kwargs,
    ) -> tuple[DownloadedFiles, ...]:
        """The Benepar model and the spaCy pipeline load_model reads.

        Takes load_model's keywords. Under nltk_data the model's directory,
        models/<benepar_model>/, where nltk.data.find finds it and where both
        of Benepar's entry points read it (benepar.integrations.downloader.
        locate_model: the spaCy component load_model adds, and benepar.Parser,
        which the constituency stage's PTB-escaped input loads on its first
        sentence): the weights, the configuration and the T5 tokenizer files
        it ships. Under site_packages the pipeline package spacy.load reads
        (spacy_doc.package_files), and the data files `import spacy` reads
        from spaCy's and thinc's own package directories
        (spacy_doc.library_files; since 25 Oct 2026). A model or package
        that is not there is declared all the same, so a lock check reports
        it missing. Nothing is loaded. Since 24 Oct 2026.
        """
        return (
            nltk_data_directory(f"models/{benepar_model}"), package_files(model_name),
            *library_files(),
        )

    def load_model(
        self,
        model_name: str = "en_core_web_sm",
        benepar_model: str = "benepar_en3",
        **kwargs,
    ) -> None:
        """Load a spaCy pipeline and add the Benepar constituency parser to it.

        Args:
            model_name:    spaCy model to use as the base pipeline, e.g. "en_core_web_sm".
                           Must be pre-installed via: python -m spacy download <model_name>
            benepar_model: Benepar constituency model to download and add to the pipeline.
                           "benepar_en3" is the default English model.

        The resulting pipeline has all standard spaCy components (tokenizer, tagger,
        parser) PLUS the Benepar component that produces constituency trees.

        Raises:
            offline.NotCachedError: During a run (offline.py), if the Benepar
                model is not in nltk_data: a run downloads nothing.
        """
        self._reject_unknown_options(kwargs)
        import benepar
        import spacy

        self._benepar_model = benepar_model
        # only download if not already present in nltk_data, and never during a run
        import nltk
        try:
            nltk.data.find(f"models/{benepar_model}")
        except LookupError as exc:
            if not downloads_allowed():
                raise not_cached(f"the Benepar model {benepar_model!r}", "nltk_data") from exc
            try:
                benepar.download(benepar_model)
            except AttributeError:
                pass  # nltk downloader bug (sha256_checksum) — model already on disk

        # give T5Tokenizer the method of transformers 4.x that benepar_en3
        # relies on, when the installed transformers lacks it
        self._patch_t5_tokenizer()

        self._nlp = spacy.load(model_name)  # base pipeline: tokenizer + tagger + parser
        self._nlp.add_pipe("benepar", config={"model": benepar_model})  # append constituency parser

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Dependency parsing and POS tagging via the underlying spaCy pipeline.

        Note: Benepar itself does not contribute to dependency or POS results.
        These come entirely from spaCy, through the pipeline SpacyAnalyzer runs
        on the same base model. For pre-tokenised input they are identical to
        SpacyAnalyzer's except on the sentences spaCy would re-segment - 65 of
        the 2077 in the EWT test split - because this wrapper pins sentence
        starts and that one does not (see `_pretokenized_doc`).

        Args:
            text: Either a raw string (spaCy tokenizes automatically) or
                  a list of pre-tokenized sentences for fair evaluation.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse with Universal POS tags, dependency heads,
        and dependency relation labels for each token.
        """
        self._ensure_loaded()

        # pre-tokenized input needs special handling to preserve token boundaries
        if isinstance(text, list):
            return self._analyze_pretokenized(text)

        doc = self._nlp(text)  # full pipeline: tokenize + tag + parse + benepar
        return self._doc_to_depparse(doc)

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Run Benepar's chart-based neural constituency parser.

        Args:
            text: Either a raw string or a list of pre-tokenized sentences.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a ConstituencyParse with exactly one bracketed tree per input
        sentence, accessed via Benepar's spaCy extension: sent._.parse_string

        Sentence boundaries are pinned for pre-tokenized input. A Doc built from a
        word list carries no boundaries, so spaCy's parser re-segments it; on the
        gold test set it split 65 of 2077 input sentences (3.1%) into two or more.
        The wrapper then returned several trees for one input and callers kept the
        first, so a partial tree covering the first few tokens was scored against a
        reference covering the whole sentence — and the sentence still counted as a
        parsing success.
        """
        self._ensure_loaded()

        if isinstance(text, list):
            if self._ptb_escaped_input:
                return self._parse_treebank_tokens(text)
            trees = []
            for words in text:
                doc = self._pretokenized_doc(words)
                # One input sentence must yield one tree. The base-class contract
                # checks this too, but failing here names the cause precisely.
                sents = list(doc.sents)
                if len(sents) != 1:
                    raise RuntimeError(
                        f"benepar: spaCy re-segmented a pre-tokenized sentence into "
                        f"{len(sents)} parts despite pinned sent_starts: {words!r}"
                    )
                trees.append(sents[0]._.parse_string)
            return ConstituencyParse(trees=trees, analyzer_name=self.name)

        doc = self._nlp(text)  # full pipeline including constituency parsing
        trees = [sent._.parse_string for sent in doc.sents]  # one tree per sentence
        return ConstituencyParse(trees=trees, analyzer_name=self.name)

    def set_ptb_escaped_input(self, enabled: bool = True) -> None:
        """Declare that constituency input arrives PTB-escaped, as a treebank writes it.

        Benepar ships two entry points and they want different input. The spaCy
        component this wrapper uses everywhere else expects ORDINARY TEXT, because
        a spaCy Doc holds what a tokenizer produced: a double quote is `"`. A
        treebank writes that same token as `` or '', a bracket as -LRB-, a slash
        as a backslash-escaped one. Benepar's own answer is `benepar.Parser` with
        `InputSentence(escaped_words=...)`, documented for "pre-tokenized datasets
        and treebanks": it runs the model over `ptb_unescape`d words and keeps the
        escaped forms as the leaves of the tree it returns.

        Handing escaped tokens to the spaCy path instead is not a small
        difference. On WSJ section 01 Benepar tagged the opening `` as -LRB- on
        210 of the roughly 305 quoted sentences and correctly only 5 times, while
        the closing '' survived. -LRB- is not in EVALB's delete set, so the gold
        tree lost its quote and Benepar's kept it; every span after the first
        quote shifted by one and the row scored 0.8421 where its unaffected
        sentences score 0.9349.

        Off by default. The EWT is web text whose quotes already are `"`, so the
        spaCy path is the right one there and those numbers stay as they were.
        """
        self._ptb_escaped_input = bool(enabled)

    def _parse_treebank_tokens(self, sentences: list[list[str]]) -> ConstituencyParse:
        """Parse PTB-escaped tokens through Benepar's own treebank interface.

        One InputSentence per input sentence, so sentence boundaries cannot move
        the way they did when a bare word list reached spaCy's parser, and the
        leaves come back as the escaped words that went in — which is what keeps
        the tree in the gold tokenisation the run is scored against.
        """
        import benepar

        if self._treebank_parser is None:
            # This loads a second copy of the weights rather than borrowing the
            # one inside the spaCy pipe: benepar.Parser owns its model and offers
            # no way to adopt an existing one, and reaching into the pipe's
            # private attribute would break on any benepar release.
            self._treebank_parser = benepar.Parser(self._benepar_model)

        inputs = [
            benepar.InputSentence(escaped_words=list(words)) for words in sentences
        ]
        trees = [
            # Escape on the Tree object, while the structure is still known, then
            # collapse: nltk pretty-prints across lines and the data model
            # promises one tree per line.
            collapse_to_single_line(str(escape_bracket_leaves_tree(tree)))
            for tree in self._treebank_parser.parse_sents(inputs)
        ]
        return ConstituencyParse(trees=trees, analyzer_name=self.name)

    def _pretokenized_doc(self, words: list[str]):
        """Build a single-sentence spaCy Doc from a word list and run the pipeline.

        `sent_starts` marks only the first token as a sentence start, which stops
        the dependency parser from re-segmenting. Without it the token boundaries
        are respected but the *sentence* boundaries are not, which is a subtler
        failure and the one that corrupted Benepar's constituency scores.

        Needed HERE and not in the spaCy wrapper, which runs the same pipeline:
        this one reads `doc.sents` and takes one tree per sentence, so a Doc split
        in two produces two trees for one input. The spaCy wrapper emits one
        Sentence per input from the word list and never consults `doc.sents`, so
        its alignment holds either way — see its `_analyze_pretokenized` for the
        attachment difference that remains and how large it is.
        """
        from spacy.tokens import Doc

        doc = Doc(
            self._nlp.vocab,
            words=list(words),
            sent_starts=[True] + [False] * (len(words) - 1) if words else [],
        )
        for _name, proc in self._nlp.pipeline:
            doc = proc(doc)
        return doc

    def _analyze_pretokenized(self, sentences: list[list[str]]) -> DependencyParse:
        """Process pre-tokenized sentences by creating spaCy Docs with preset tokens.

        Each word list becomes a spaCy Doc with fixed token boundaries,
        bypassing spaCy's tokenizer. All other pipeline components (tagger,
        dependency parser, Benepar) are then run on each Doc.

        The tokens are built by `spacy_doc.doc_tokens`, so the relation labels
        go through the same SD->UD conversion spaCy's own wrapper applies.
        Benepar's dependency output comes from the underlying spaCy model, so
        leaving it raw meant the two wrappers disagreed on identical parses
        whenever mapping was enabled.

        Args:
            sentences: List of word lists, one per sentence.
        """
        result_sentences = []
        for words in sentences:
            doc = self._pretokenized_doc(words)
            tokens = doc_tokens(doc, tier=self._map_tier)
            result_sentences.append(Sentence(text=" ".join(words), tokens=tokens))
        return DependencyParse(sentences=result_sentences, analyzer_name=self.name)

    def _doc_to_depparse(self, doc) -> DependencyParse:
        """Convert a spaCy Doc (with multiple sentences) to our DependencyParse format.

        Handles sentence-relative indexing: spaCy token indices are document-global,
        but our Token.index must be 1-based within each sentence.

        Args:
            doc: A processed spaCy Doc object.
        """
        return doc_to_depparse(doc, tier=self._map_tier, name=self.name)

    @staticmethod
    def _patch_t5_tokenizer():
        """Give T5Tokenizer the build_inputs_with_special_tokens of transformers 4.x.

        benepar_en3 was saved under transformers 4.x (its config.json records
        4.2.2), whose T5 tokenizer has this method; transformers 5 (5.3 here)
        does not, and without it the model does not load. Benepar calls it
        once, on [-100], when it builds the parser (benepar/retokenization.py),
        to learn where the tokenizer puts its special tokens: the position of
        the end token </s> is where it then reads the vector that stands for
        the end of the sentence. The method added here therefore returns what
        transformers 4.57.6's T5TokenizerFast returns - the class benepar gets
        under 4.x, since it requires a fast tokenizer: the ids followed by
        </s>, and for a pair each list followed by </s>, with the end token's
        id taken from the tokenizer itself. Checked on the first 100 sentences
        of WSJ 01: all 100 trees are those benepar_en3 gives under transformers
        4.57.6, where this patch does not apply.

        Until 24 Oct 2026 the method returned the ids unchanged. Benepar then
        found no end token after the sentence and took the end-of-sentence
        vector from position 0, where it also takes the start vector; the
        tokens the model read were the same. One of those 100 trees differed
        (wsj_0102.mrg, sentence 15).

        Nothing is added when the installed transformers has the method.
        """
        try:
            from transformers import T5Tokenizer
        except ImportError:
            return
        if hasattr(T5Tokenizer, "build_inputs_with_special_tokens"):
            return

        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            # T5TokenizerFast.build_inputs_with_special_tokens of transformers
            # 4.57.6, without its prefix_tokens: a class attribute that is []
            # for T5 and that nothing sets.
            token_ids_0 = token_ids_0 + [self.eos_token_id]
            if token_ids_1 is None:
                return token_ids_0
            token_ids_1 = token_ids_1 + [self.eos_token_id]
            return token_ids_0 + token_ids_1

        T5Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    def _ensure_loaded(self) -> None:
        """Verify that load_model() has been called before analysis.

        Raises:
            RuntimeError: If the spaCy pipeline has not been loaded.
        """
        self._require_loaded(self._nlp is not None)  # guard against usage before load_model()
