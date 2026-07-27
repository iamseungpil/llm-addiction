"""Bathina et al. (2021) cognitive distortion schemata (CDS): the real 241 n-grams.

PROVENANCE — this file contains the authors' published lexicon verbatim. Nothing
here was written, paraphrased, extended or guessed by us.

  Paper   Bathina, K. C., ten Thij, M., Lorenzo-Luaces, L., Rutter, L. A. &
          Bollen, J. (2021). "Individuals with depression express more distorted
          thinking on social media." Nature Human Behaviour 5(4), 458-466.
          doi:10.1038/s41562-021-01050-7  (preprint: arXiv:2002.02800)

  Source  https://github.com/mctenthij/CDS_paper  ->  data/list_of_CDS.tsv
          (the repository named in the paper's own Data availability and Code
          availability statements)
          raw: https://raw.githubusercontent.com/mctenthij/CDS_paper/master/data/list_of_CDS.tsv
          sha256 918eda112c43e27bf3a3020bb2135887edbc80575324af3b6a28a96c6bb32da9
          retrieved 2026-07-27

  Mirror  https://github.com/aedinger7/distortion_polarization
          -> CDS/translations/list_of_CDS_EN.tsv  is byte-identical to the above
          (verified by diff). That repo also carries __CDS__.py, the matching
          code reproduced as `contains_cds` / `annotate` below, and NL/DE/ES
          translations of the same lexicon.

  Verified  Row counts per category reproduce Table 4 of the paper exactly
            (Personalizing 14, Emotional reasoning 7, Overgeneralizing 21,
            Mental filtering 14, Disqualifying the positive 14, Labelling and
            mislabelling 44, Dichotomous reasoning 23, Fortune-telling 8,
            Magnification and minimization 8, Should statements 5,
            Mindreading 72, Catastrophizing 11; total 241).

NOT OBTAINED — Lalk et al. (2024), "Depression Symptoms are Associated with
Frequency of Cognitive Distortions in Psychotherapy Transcripts", Cognitive
Therapy and Research (doi:10.1007/s10608-024-10542-5), which modifies this list
into 14 German categories, is paywalled at Springer; no OSF/GitHub deposit for
their modified n-grams was found, they are not in EuropePMC, and the paper's
supplement could not be reached. Their 14 categories and German n-grams are
therefore ABSENT from this file. Do not cite a 14-category variant from here.

UPSTREAM DEFECTS, preserved verbatim (do not silently "fix" — patch at use site
if you must, and say so in the paper):
  * Mindreading / "we will not believe" carries the variants
    ["he won't believe", "he wont believe"] — a copy-paste error upstream.
  * Should statements / "ought" carries ["oughn't", "oughnt"] — misspelling of
    "oughtn't".
  * Fortune-telling / "we will not " has a trailing space, so its \b-delimited
    regex behaves differently from its seven siblings.

LICENSE — the upstream repository ships no LICENSE file. Treat the lexicon as
the authors' published research material: cite it, do not relicense it.
"""

from __future__ import annotations

import re
from collections import OrderedDict

CITATION = (
    "Bathina, K. C., ten Thij, M., Lorenzo-Luaces, L., Rutter, L. A. & Bollen, J. "
    "(2021). Individuals with depression express more distorted thinking on social "
    "media. Nature Human Behaviour 5(4), 458-466. doi:10.1038/s41562-021-01050-7"
)
SOURCE_URL = "https://github.com/mctenthij/CDS_paper/blob/master/data/list_of_CDS.tsv"
SOURCE_SHA256 = "918eda112c43e27bf3a3020bb2135887edbc80575324af3b6a28a96c6bb32da9"

# The 12 categories, in the order they appear upstream.
CATEGORIES: tuple[str, ...] = (
    'Labeling and mislabeling',
    'Catastrophizing',
    'Dichotomous Reasoning',
    'Emotional Reasoning',
    'Disqualifying the Positive',
    'Magnification and Minimization',
    'Mental Filtering',
    'Mindreading',
    'Fortune-telling',
    'Overgeneralizing',
    'Personalizing',
    'Should statements',
)

# The 241 CDS n-grams, verbatim from list_of_CDS.tsv, in file order.
# Each entry is (category, marker, variants). `variants` are alternative
# spellings (contractions) that count as the same schema.
CDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ('Labeling and mislabeling', 'I am a', ("I'm a", 'Im a')),
    ('Labeling and mislabeling', 'he is a', ("he's a", 'hes a')),
    ('Labeling and mislabeling', 'she is a', ("she's a", 'shes a')),
    ('Labeling and mislabeling', 'they are a', ("they're a", 'theyre a')),
    ('Labeling and mislabeling', 'it is a', ("it's a", 'its a')),
    ('Labeling and mislabeling', 'that is a', ("that's a", 'thats a')),
    ('Labeling and mislabeling', 'sucks at', ()),
    ('Labeling and mislabeling', 'suck at', ()),
    ('Labeling and mislabeling', 'I never', ()),
    ('Labeling and mislabeling', 'he never', ()),
    ('Labeling and mislabeling', 'she never', ()),
    ('Labeling and mislabeling', 'you never', ()),
    ('Labeling and mislabeling', 'we never', ()),
    ('Labeling and mislabeling', 'they never', ()),
    ('Labeling and mislabeling', 'I am an', ("I'm an", 'Im an')),
    ('Labeling and mislabeling', 'he is an', ("he's an", 'hes an')),
    ('Labeling and mislabeling', 'she is an', ("she's an", 'shes an')),
    ('Labeling and mislabeling', 'they are an', ("they're an", 'theyre an')),
    ('Labeling and mislabeling', 'it is an', ("it's an", 'its an')),
    ('Labeling and mislabeling', 'that is an', ("that's an", 'thats an')),
    ('Labeling and mislabeling', 'a burden', ()),
    ('Labeling and mislabeling', 'a complete', ()),
    ('Labeling and mislabeling', 'a completely', ()),
    ('Labeling and mislabeling', 'a huge', ()),
    ('Labeling and mislabeling', 'a loser', ()),
    ('Labeling and mislabeling', 'a major', ()),
    ('Labeling and mislabeling', 'a total', ()),
    ('Labeling and mislabeling', 'a totally', ()),
    ('Labeling and mislabeling', 'a weak', ()),
    ('Labeling and mislabeling', 'an absolute', ()),
    ('Labeling and mislabeling', 'an utter', ()),
    ('Labeling and mislabeling', 'a bad', ()),
    ('Labeling and mislabeling', 'a broken', ()),
    ('Labeling and mislabeling', 'a damaged', ()),
    ('Labeling and mislabeling', 'a helpless', ()),
    ('Labeling and mislabeling', 'a hopeless', ()),
    ('Labeling and mislabeling', 'an incompetent', ()),
    ('Labeling and mislabeling', 'a toxic', ()),
    ('Labeling and mislabeling', 'an ugly', ()),
    ('Labeling and mislabeling', 'an undesirable', ()),
    ('Labeling and mislabeling', 'an unlovable', ()),
    ('Labeling and mislabeling', 'a worthless', ()),
    ('Labeling and mislabeling', 'a horrible', ()),
    ('Labeling and mislabeling', 'a terrible', ()),
    ('Catastrophizing', 'will fail', ("'ll fail",)),
    ('Catastrophizing', 'will go wrong', ("'ll go wrong",)),
    ('Catastrophizing', 'will end', ("'ll end",)),
    ('Catastrophizing', 'will be impossible', ("'ll be impossible",)),
    ('Catastrophizing', 'will not happen', ("won't happen", 'wont happen')),
    ('Catastrophizing', 'will be terrible', ("'ll be terrible",)),
    ('Catastrophizing', 'will be horrible', ("'ll be horrible",)),
    ('Catastrophizing', 'will be a catastrophe', ("'ll be a catastrophe",)),
    ('Catastrophizing', 'will be a disaster', ("'ll be a disaster",)),
    ('Catastrophizing', 'will never end', ()),
    ('Catastrophizing', 'will not end', ("won't end", 'wont end')),
    ('Dichotomous Reasoning', 'only', ()),
    ('Dichotomous Reasoning', 'every', ()),
    ('Dichotomous Reasoning', 'everyone', ()),
    ('Dichotomous Reasoning', 'everybody', ()),
    ('Dichotomous Reasoning', 'everything', ()),
    ('Dichotomous Reasoning', 'everywhere', ()),
    ('Dichotomous Reasoning', 'always', ()),
    ('Dichotomous Reasoning', 'perfect', ()),
    ('Dichotomous Reasoning', 'the best', ()),
    ('Dichotomous Reasoning', 'all', ()),
    ('Dichotomous Reasoning', 'not a single', ("'t a single",)),
    ('Dichotomous Reasoning', 'no one', ()),
    ('Dichotomous Reasoning', 'nobody', ()),
    ('Dichotomous Reasoning', 'nothing', ()),
    ('Dichotomous Reasoning', 'nowhere', ()),
    ('Dichotomous Reasoning', 'never', ()),
    ('Dichotomous Reasoning', 'worthless', ()),
    ('Dichotomous Reasoning', 'the worst', ()),
    ('Dichotomous Reasoning', 'neither', ()),
    ('Dichotomous Reasoning', 'nor', ()),
    ('Dichotomous Reasoning', 'either or', ()),
    ('Dichotomous Reasoning', 'black or white', ()),
    ('Dichotomous Reasoning', 'ever', ()),
    ('Emotional Reasoning', 'but I feel', ()),
    ('Emotional Reasoning', 'since I feel', ()),
    ('Emotional Reasoning', 'because I feel', ()),
    ('Emotional Reasoning', 'but it feels', ()),
    ('Emotional Reasoning', 'since it feels', ()),
    ('Emotional Reasoning', 'because it feels', ()),
    ('Emotional Reasoning', 'still feels', ()),
    ('Disqualifying the Positive', 'great but', ()),
    ('Disqualifying the Positive', 'good but', ()),
    ('Disqualifying the Positive', 'OK but', ()),
    ('Disqualifying the Positive', 'not that great', ("'t that great",)),
    ('Disqualifying the Positive', 'not that good', ("'t that good",)),
    ('Disqualifying the Positive', 'it was not', ("it wasn't", 'it wasnt')),
    ('Disqualifying the Positive', 'not all that', ("'t all that",)),
    ('Disqualifying the Positive', 'fine but', ()),
    ('Disqualifying the Positive', 'acceptable but', ()),
    ('Disqualifying the Positive', 'great yet', ()),
    ('Disqualifying the Positive', 'good yet', ()),
    ('Disqualifying the Positive', 'OK yet', ()),
    ('Disqualifying the Positive', 'fine yet', ()),
    ('Disqualifying the Positive', 'acceptable yet', ()),
    ('Magnification and Minimization', 'worst', ()),
    ('Magnification and Minimization', 'best', ()),
    ('Magnification and Minimization', 'not important', ("'t important",)),
    ('Magnification and Minimization', 'not count', ("'t count",)),
    ('Magnification and Minimization', 'not matter', ("'t matter",)),
    ('Magnification and Minimization', 'no matter', ()),
    ('Magnification and Minimization', 'the only thing', ()),
    ('Magnification and Minimization', 'the one thing', ()),
    ('Mental Filtering', 'I see only', ()),
    ('Mental Filtering', 'all I see', ()),
    ('Mental Filtering', 'all I can see', ()),
    ('Mental Filtering', 'can only think', ()),
    ('Mental Filtering', 'nothing good', ()),
    ('Mental Filtering', 'nothing right', ()),
    ('Mental Filtering', 'completely bad', ()),
    ('Mental Filtering', 'completely wrong', ()),
    ('Mental Filtering', 'only the bad', ()),
    ('Mental Filtering', 'only the worst', ()),
    ('Mental Filtering', 'if I just', ()),
    ('Mental Filtering', 'if I only', ()),
    ('Mental Filtering', 'if it just', ()),
    ('Mental Filtering', 'if it only', ()),
    ('Mindreading', 'everyone believes', ()),
    ('Mindreading', 'everyone knows', ()),
    ('Mindreading', 'everyone thinks', ()),
    ('Mindreading', 'everyone will believe', ("everyone'll believe",)),
    ('Mindreading', 'everyone will know', ("everyone'll know",)),
    ('Mindreading', 'everyone will think', ("everyone'll think",)),
    ('Mindreading', 'nobody believes', ()),
    ('Mindreading', 'nobody knows', ()),
    ('Mindreading', 'nobody thinks', ()),
    ('Mindreading', 'nobody will believe', ("nobody'll believe",)),
    ('Mindreading', 'nobody will know', ("nobody'll know",)),
    ('Mindreading', 'nobody will think', ("nobody'll think",)),
    ('Mindreading', 'he believes', ()),
    ('Mindreading', 'he knows', ()),
    ('Mindreading', 'he thinks', ()),
    ('Mindreading', 'he does not believe', ("he doesn't believe", 'he doesnt believe')),
    ('Mindreading', 'he does not know', ("he doesn't know", 'he doesnt know')),
    ('Mindreading', 'he does not think', ("he doesn't think", 'he doesnt think')),
    ('Mindreading', 'he will believe', ("he'll believe",)),
    ('Mindreading', 'he will know', ("he'll know",)),
    ('Mindreading', 'he will think', ("he'll think",)),
    ('Mindreading', 'he will not believe', ("he won't believe", 'he wont believe')),
    ('Mindreading', 'he will not know', ("he won't know", 'he wont know')),
    ('Mindreading', 'he will not think', ("he won't think", 'he wont think')),
    ('Mindreading', 'she believes', ()),
    ('Mindreading', 'she knows', ()),
    ('Mindreading', 'she thinks', ()),
    ('Mindreading', 'she does not believe', ("she doesn't believe", 'she doesnt believe')),
    ('Mindreading', 'she does not know', ("she doesn't know", 'she doesnt know')),
    ('Mindreading', 'she does not think', ("she doesn't think", 'she doesnt think')),
    ('Mindreading', 'she will believe', ("she'll believe",)),
    ('Mindreading', 'she will know', ("she'll know",)),
    ('Mindreading', 'she will think', ("she'll think",)),
    ('Mindreading', 'she will not believe', ("she won't believe", 'she wont believe')),
    ('Mindreading', 'she will not know', ("she won't know", 'she wont know')),
    ('Mindreading', 'she will not think', ("she won't think", 'she wont think')),
    ('Mindreading', 'they believe', ()),
    ('Mindreading', 'they know', ()),
    ('Mindreading', 'they think', ()),
    ('Mindreading', 'they do not believe', ("they don't believe", 'they dont believe')),
    ('Mindreading', 'they do not know', ("they don't know", 'they dont know')),
    ('Mindreading', 'they do not think', ("they don't think", 'they dont think')),
    ('Mindreading', 'they will believe', ("they'll believe", 'theyll believe')),
    ('Mindreading', 'they will know', ("they'll know", 'theyll know')),
    ('Mindreading', 'they will think', ("they'll think", 'theyll think')),
    ('Mindreading', 'they will not believe', ("they won't believe", 'they wont believe')),
    ('Mindreading', 'they will not know', ("they won't know", 'they wont know')),
    ('Mindreading', 'they will not think', ("they won't think", 'they wont think')),
    ('Mindreading', 'we believe', ()),
    ('Mindreading', 'we know', ()),
    ('Mindreading', 'we think', ()),
    ('Mindreading', 'we do not believe', ("we don't believe", 'we dont believe')),
    ('Mindreading', 'we do not know', ("we don't know", 'we dont know')),
    ('Mindreading', 'we do not think', ("we don't think", 'we dont think')),
    ('Mindreading', 'we will believe', ("we'll believe",)),
    ('Mindreading', 'we will know', ("we'll know",)),
    ('Mindreading', 'we will think', ("we'll think",)),
    ('Mindreading', 'we will not believe', ("he won't believe", 'he wont believe')),
    ('Mindreading', 'we will not know', ("we won't know", 'we wont know')),
    ('Mindreading', 'we will not think', ("we won't think", 'we wont think')),
    ('Mindreading', 'you believe', ()),
    ('Mindreading', 'you know', ()),
    ('Mindreading', 'you think', ()),
    ('Mindreading', 'you do not believe', ("you don't believe", 'you dont believe')),
    ('Mindreading', 'you do not know', ("you don't know", 'you dont know')),
    ('Mindreading', 'you do not think', ("you don't think", 'you dont think')),
    ('Mindreading', 'you will believe', ("you'll believe", 'youll believe')),
    ('Mindreading', 'you will know', ("you'll know", 'youll know')),
    ('Mindreading', 'you will think', ("you'll think", 'youll think')),
    ('Mindreading', 'you will not believe', ("you won't believe", 'you wont believe')),
    ('Mindreading', 'you will not know', ("you won't know", 'you wont know')),
    ('Mindreading', 'you will not think', ("you won't think", 'you wont think')),
    ('Fortune-telling', 'I will not', ("I won't", 'I wont')),
    ('Fortune-telling', 'we will not ', ("we won't", 'we wont')),
    ('Fortune-telling', 'you will not', ("you won't", 'you wont')),
    ('Fortune-telling', 'they will not', ("they won't", 'they wont')),
    ('Fortune-telling', 'it will not', ("it won't", 'it wont')),
    ('Fortune-telling', 'that will not', ("that won't", 'that wont')),
    ('Fortune-telling', 'he will not', ("he won't", 'he wont')),
    ('Fortune-telling', 'she will not', ("she won't", 'she wont')),
    ('Overgeneralizing', 'all of the time', ()),
    ('Overgeneralizing', 'all of them', ()),
    ('Overgeneralizing', 'all the time', ()),
    ('Overgeneralizing', 'always happens', ()),
    ('Overgeneralizing', 'always like', ()),
    ('Overgeneralizing', 'happens every time', ()),
    ('Overgeneralizing', 'completely', ()),
    ('Overgeneralizing', 'no one ever', ()),
    ('Overgeneralizing', 'nobody ever', ()),
    ('Overgeneralizing', 'every single one of them', ()),
    ('Overgeneralizing', 'every single one of you', ()),
    ('Overgeneralizing', 'I always', ()),
    ('Overgeneralizing', 'you always', ()),
    ('Overgeneralizing', 'he always', ()),
    ('Overgeneralizing', 'she always', ()),
    ('Overgeneralizing', 'they always', ()),
    ('Overgeneralizing', 'I am always', ("I'm always", 'Im always')),
    ('Overgeneralizing', 'you are always', ("you're always", 'youre always')),
    ('Overgeneralizing', 'he is always', ("he's always", 'hes always')),
    ('Overgeneralizing', 'she is always', ("she's always", 'shes always')),
    ('Overgeneralizing', 'they are always', ("they're always", 'theyre always')),
    ('Personalizing', 'all me', ()),
    ('Personalizing', 'all my', ()),
    ('Personalizing', 'because I', ()),
    ('Personalizing', 'because my', ()),
    ('Personalizing', 'because of my', ()),
    ('Personalizing', 'because of me', ()),
    ('Personalizing', 'I am responsible', ("I'm responsible", 'Im responsible')),
    ('Personalizing', 'blame me', ()),
    ('Personalizing', 'I caused', ()),
    ('Personalizing', 'I feel responsible', ()),
    ('Personalizing', 'all my doing', ()),
    ('Personalizing', 'all my fault', ()),
    ('Personalizing', 'my bad', ()),
    ('Personalizing', 'my responsibility', ()),
    ('Should statements', 'should', ("shouldn't", 'shouldnt')),
    ('Should statements', 'ought', ("oughn't", 'oughnt')),
    ('Should statements', 'must', ("mustn't", 'mustnt')),
    ('Should statements', 'have to', ()),
    ('Should statements', 'has to', ()),
)

assert len(CDS) == 241

BY_CATEGORY: "OrderedDict[str, list[tuple[str, tuple[str, ...]]]]" = OrderedDict(
    (c, [(m, v) for cat, m, v in CDS if cat == c]) for c in CATEGORIES
)

# ---------------------------------------------------------------------------
# Per-n-gram between-cohort prevalence ratio (depressed cohort D vs random R),
# DERIVED BY US from the authors' own published bootstrap output
# https://raw.githubusercontent.com/mctenthij/CDS_paper/master/bootstrap/relative_prevalence_phrase.tsv
# (10,000 bootstrap runs). Value is (median, p2.5, p97.5) with inf runs dropped.
# PR > 1 means the n-gram is more prevalent in the depressed cohort.
# 237 of 241 n-grams have a finite value. Four are absent because every
# bootstrap run was blank/infinite (zero occurrences in the R cohort):
# 'will be a catastrophe', 'acceptable yet', 'everyone will believe',
# 'they will not think'.
# These numbers are a summary we computed, not a table printed in the paper —
# label them as such if reported.
PR_PHRASE: dict[str, tuple[float, float, float]] = {
    'I am a': (1.7312, 1.618, 1.8543),
    'he is a': (0.8093, 0.7284, 0.8973),
    'she is a': (1.0406, 0.9231, 1.1732),
    'they are a': (1.1736, 1.0159, 1.3446),
    'it is a': (1.1323, 1.0701, 1.1979),
    'that is a': (0.9746, 0.9038, 1.0465),
    'sucks at': (1.1805, 0.6996, 1.7949),
    'suck at': (1.428, 1.1679, 1.7217),
    'I never': (1.5045, 1.4067, 1.6077),
    'he never': (0.9163, 0.7591, 1.1088),
    'she never': (1.2404, 0.9532, 1.5733),
    'you never': (1.2303, 1.1123, 1.3541),
    'we never': (1.0772, 0.9159, 1.2604),
    'they never': (1.0512, 0.9094, 1.2043),
    'I am an': (1.8187, 1.62, 2.0225),
    'he is an': (0.7747, 0.6348, 0.9317),
    'she is an': (1.2524, 1.0298, 1.5171),
    'they are an': (1.3387, 0.9391, 1.8333),
    'it is an': (1.1503, 1.0454, 1.2664),
    'that is an': (1.1292, 0.9807, 1.2995),
    'a burden': (3.1849, 2.2919, 4.3449),
    'a complete': (1.0398, 0.8791, 1.2224),
    'a completely': (1.1489, 0.9192, 1.4026),
    'a huge': (1.2979, 1.1787, 1.4333),
    'a loser': (0.8539, 0.6372, 1.1383),
    'a major': (0.78, 0.6538, 0.9265),
    'a total': (1.1615, 0.947, 1.4536),
    'a totally': (1.0018, 0.7522, 1.2955),
    'a weak': (0.7092, 0.5132, 0.9583),
    'an absolute': (1.1993, 0.9366, 1.5366),
    'an utter': (1.4317, 0.8384, 2.258),
    'a bad': (1.2077, 1.1282, 1.2923),
    'a broken': (1.4054, 1.1822, 1.6673),
    'a damaged': (2.1367, 0.9647, 4.2103),
    'a helpless': (1.9376, 0.7746, 3.9343),
    'a hopeless': (1.889, 1.2336, 2.7959),
    'an incompetent': (0.3796, 0.1318, 0.7929),
    'a toxic': (1.4392, 0.939, 2.0711),
    'an ugly': (1.2456, 0.9795, 1.5366),
    'an undesirable': (3.4846, 0.5194, 17.1984),
    'an unlovable': (1.5739, 0.0, 13.0558),
    'a worthless': (1.0045, 0.5749, 1.6215),
    'a horrible': (1.4661, 1.2377, 1.7214),
    'a terrible': (1.1174, 0.9623, 1.2907),
    'will fail': (0.9071, 0.5954, 1.3195),
    'will go wrong': (3.7374, 1.2206, 9.6755),
    'will end': (0.9464, 0.6993, 1.1971),
    'will be impossible': (0.6417, 0.0, 1.7736),
    'will not happen': (0.8574, 0.6736, 1.0595),
    'will be terrible': (0.0, 0.0, 0.0),
    'will be horrible': (0.0, 0.0, 0.0),
    'will be a disaster': (0.6474, 0.0, 2.617),
    'will never end': (1.053, 0.5227, 1.8079),
    'will not end': (0.8638, 0.5712, 1.2248),
    'only': (1.1239, 1.0727, 1.1751),
    'every': (1.1094, 1.0559, 1.1662),
    'everyone': (1.3208, 1.2487, 1.3973),
    'everybody': (0.7658, 0.6507, 0.901),
    'everything': (1.2784, 1.2058, 1.3505),
    'everywhere': (1.1971, 1.0927, 1.3158),
    'always': (1.2754, 1.2092, 1.3445),
    'perfect': (1.0569, 0.9693, 1.1542),
    'the best': (1.0557, 0.9958, 1.12),
    'all': (1.0856, 1.0426, 1.1277),
    'not a single': (1.0035, 0.7752, 1.2777),
    'no one': (1.215, 1.125, 1.3132),
    'nobody': (0.8457, 0.7493, 0.9498),
    'nothing': (1.0715, 1.012, 1.1325),
    'nowhere': (1.1673, 1.0406, 1.3059),
    'never': (1.2173, 1.1692, 1.2661),
    'worthless': (1.4727, 1.1804, 1.8076),
    'the worst': (1.258, 1.1708, 1.3494),
    'neither': (1.0441, 0.9296, 1.1677),
    'nor': (0.9384, 0.7804, 1.1177),
    'either or': (0.8019, 0.4307, 1.2975),
    'black or white': (0.5688, 0.2496, 1.1416),
    'ever': (1.2707, 1.2164, 1.3256),
    'but I feel': (1.8622, 1.619, 2.1347),
    'since I feel': (2.2024, 0.0, 8.1516),
    'because I feel': (2.935, 2.2283, 3.8377),
    'but it feels': (1.9127, 1.4479, 2.483),
    'since it feels': (3.8164, 0.9979, 13.053),
    'because it feels': (1.62, 0.909, 2.6493),
    'still feels': (1.6757, 1.2029, 2.2709),
    'great but': (1.0838, 0.8663, 1.3331),
    'good but': (0.983, 0.8282, 1.1594),
    'OK but': (1.4344, 1.1126, 1.8347),
    'not that great': (1.8599, 1.295, 2.5579),
    'not that good': (1.0452, 0.7484, 1.3993),
    'it was not': (1.2531, 1.1574, 1.3586),
    'not all that': (0.8931, 0.6448, 1.1902),
    'fine but': (1.2935, 0.8007, 1.8935),
    'acceptable but': (0.8481, 0.0, 2.5542),
    'great yet': (0.6984, 0.0, 4.2011),
    'good yet': (2.2576, 0.897, 4.67),
    'OK yet': (0.0, 0.0, 0.0),
    'fine yet': (1.1634, 0.0, 9.0721),
    'worst': (1.1599, 1.0881, 1.2362),
    'best': (1.0434, 0.9931, 1.0956),
    'not important': (1.1352, 0.703, 1.688),
    'not count': (0.8896, 0.7431, 1.0587),
    'not matter': (1.1118, 0.9771, 1.2609),
    'no matter': (1.3848, 1.1789, 1.5791),
    'the only thing': (1.354, 1.2339, 1.4858),
    'the one thing': (1.0583, 0.7862, 1.3685),
    'I see only': (0.7538, 0.0, 2.7889),
    'all I see': (0.9283, 0.7177, 1.1789),
    'all I can see': (1.6635, 0.8793, 2.8219),
    'can only think': (1.0486, 0.5095, 1.8734),
    'nothing good': (1.2837, 0.8771, 1.81),
    'nothing right': (0.9309, 0.1765, 2.2262),
    'completely bad': (0.0, 0.0, 0.0),
    'completely wrong': (1.0091, 0.6313, 1.4756),
    'only the bad': (2.7675, 0.0, 17.4325),
    'only the worst': (0.0, 0.0, 0.0),
    'if I just': (1.7454, 1.3353, 2.2522),
    'if I only': (2.3956, 1.5103, 3.712),
    'if it just': (1.2552, 0.3525, 2.9166),
    'if it only': (2.7578, 1.1077, 5.7909),
    'everyone believes': (1.0057, 0.2208, 2.328),
    'everyone knows': (0.9477, 0.7855, 1.136),
    'everyone thinks': (1.4289, 0.8333, 2.2231),
    'everyone will know': (0.9072, 0.1902, 2.1668),
    'everyone will think': (5.6977, 1.1143, 26.735),
    'nobody believes': (0.4216, 0.1266, 0.8849),
    'nobody knows': (0.9349, 0.6709, 1.254),
    'nobody thinks': (0.5591, 0.0, 1.5809),
    'nobody will believe': (0.0, 0.0, 0.0),
    'nobody will know': (0.6891, 0.0, 2.7873),
    'nobody will think': (1.5437, 0.0, 13.0628),
    'he believes': (0.4726, 0.2852, 0.723),
    'he knows': (0.9107, 0.7474, 1.1387),
    'he thinks': (0.889, 0.7086, 1.1453),
    'he does not believe': (0.8279, 0.3915, 1.449),
    'he does not know': (0.9412, 0.6606, 1.3537),
    'he does not think': (0.3367, 0.0961, 0.7035),
    'he will believe': (8.976, 0.0, 27.4921),
    'he will know': (1.3712, 0.4846, 2.8501),
    'he will think': (1.7833, 0.6533, 3.8784),
    'he will not believe': (0.0, 0.0, 0.0),
    'he will not know': (1.7849, 0.3133, 5.472),
    'he will not think': (1.5251, 0.0, 12.8261),
    'she believes': (1.0496, 0.599, 1.7049),
    'she knows': (1.1453, 0.9392, 1.3788),
    'she thinks': (1.3069, 1.029, 1.6185),
    'she does not believe': (1.752, 0.7188, 3.4712),
    'she does not know': (1.4373, 1.0218, 1.9597),
    'she does not think': (1.1398, 0.4504, 2.2252),
    'she will believe': (0.0, 0.0, 0.0),
    'she will know': (0.7283, 0.1444, 1.9098),
    'she will think': (1.1389, 0.0, 4.6336),
    'she will not believe': (0.0, 0.0, 0.0),
    'she will not know': (1.6486, 0.0, 8.9827),
    'she will not think': (0.0, 0.0, 0.0),
    'they believe': (0.7598, 0.5499, 1.0116),
    'they know': (0.9717, 0.8693, 1.0855),
    'they think': (1.1511, 1.0085, 1.3119),
    'they do not believe': (0.7794, 0.4322, 1.231),
    'they do not know': (1.1931, 0.9948, 1.4107),
    'they do not think': (1.2062, 0.7267, 1.8217),
    'they will believe': (0.723, 0.1262, 1.8479),
    'they will know': (1.4327, 0.8051, 2.3554),
    'they will think': (1.7379, 0.8457, 3.0845),
    'they will not believe': (2.2093, 0.0, 12.9834),
    'they will not know': (1.9978, 0.7632, 4.3031),
    'we believe': (0.5253, 0.3546, 0.7524),
    'we know': (0.6711, 0.5886, 0.7656),
    'we think': (1.0561, 0.8747, 1.2696),
    'we do not believe': (0.3449, 0.1443, 0.6634),
    'we do not know': (0.9556, 0.7638, 1.1762),
    'we do not think': (1.0789, 0.4767, 1.9785),
    'we will believe': (0.7698, 0.0, 2.7878),
    'we will know': (0.5882, 0.3253, 0.9435),
    'we will think': (0.5041, 0.0, 2.7752),
    'we will not believe': (0.0, 0.0, 0.0),
    'we will not know': (0.7861, 0.2948, 1.5066),
    'we will not think': (0.0, 0.0, 0.0),
    'you believe': (1.0576, 0.8295, 1.2831),
    'you know': (1.2005, 1.1325, 1.2729),
    'you think': (0.9978, 0.8847, 1.1109),
    'you do not believe': (1.2043, 0.8883, 1.581),
    'you do not know': (1.2302, 1.0765, 1.3891),
    'you do not think': (0.9322, 0.7006, 1.1936),
    'you will believe': (0.9925, 0.3921, 1.9301),
    'you will know': (1.1552, 0.8744, 1.5039),
    'you will think': (0.7163, 0.3125, 1.2896),
    'you will not believe': (0.0636, 0.0146, 0.7339),
    'you will not know': (0.995, 0.3909, 1.9879),
    'you will not think': (0.652, 0.0, 4.4531),
    'I will not': (1.5113, 1.4071, 1.6226),
    'we will not ': (0.7645, 0.6334, 0.92),
    'you will not': (0.7819, 0.4775, 1.1226),
    'they will not': (0.9261, 0.8195, 1.044),
    'it will not': (1.173, 1.0549, 1.2963),
    'that will not': (1.0738, 0.8955, 1.2703),
    'he will not': (0.8098, 0.6852, 0.9558),
    'she will not': (1.0017, 0.7232, 1.3015),
    'all of the time': (1.1785, 0.6902, 1.8537),
    'all of them': (1.1556, 1.0328, 1.2918),
    'all the time': (1.5496, 1.4301, 1.6766),
    'always happens': (1.1102, 0.6887, 1.6734),
    'always like': (1.0549, 0.7996, 1.3434),
    'happens every time': (0.6522, 0.2654, 1.2081),
    'completely': (1.3458, 1.2331, 1.4699),
    'no one ever': (1.1447, 0.921, 1.3968),
    'nobody ever': (1.2388, 0.8588, 1.7102),
    'every single one of them': (1.7759, 1.0715, 2.7776),
    'every single one of you': (2.3087, 1.2817, 3.8353),
    'I always': (1.5831, 1.4678, 1.706),
    'you always': (1.2126, 1.0615, 1.3763),
    'he always': (1.3272, 1.0696, 1.6605),
    'she always': (1.4164, 1.0042, 1.8941),
    'they always': (1.0983, 0.9505, 1.2578),
    'I am always': (1.9612, 1.6998, 2.3429),
    'you are always': (1.384, 1.1114, 1.6894),
    'he is always': (0.9677, 0.7689, 1.1974),
    'she is always': (1.5472, 1.0788, 2.1396),
    'they are always': (1.1098, 0.841, 1.4579),
    'all me': (1.17, 0.7217, 1.7603),
    'all my': (1.5487, 1.4329, 1.6707),
    'because I': (2.395, 2.1914, 2.6188),
    'because my': (2.8095, 2.4577, 3.2124),
    'because of my': (3.3683, 2.7662, 4.0674),
    'because of me': (1.8257, 1.0828, 2.871),
    'I am responsible': (1.8753, 0.8493, 3.8477),
    'blame me': (1.7616, 1.2918, 2.3476),
    'I caused': (3.7025, 1.2954, 9.2884),
    'I feel responsible': (0.7678, 0.0, 4.5814),
    'all my doing': (0.0, 0.0, 0.0),
    'all my fault': (2.98, 1.634, 5.1783),
    'my bad': (0.9724, 0.8371, 1.1272),
    'my responsibility': (1.4762, 0.6981, 2.7367),
    'should': (1.0026, 0.9543, 1.0538),
    'ought': (0.7934, 0.5075, 1.1883),
    'must': (0.8714, 0.8103, 0.9367),
    'have to': (1.2998, 1.2192, 1.3768),
    'has to': (0.8979, 0.8265, 0.9729),
}


# ---------------------------------------------------------------------------
# Matching. This reproduces find_CDS_in_text / process_dataset from the authors'
# __CDS__.py (https://github.com/aedinger7/distortion_polarization/blob/main/__CDS__.py)
# so our numbers are comparable to theirs. Two properties of their design that
# are easy to get wrong and that we keep on purpose:
#   1. the marker is word-boundary matched (\b...\b) against LOWERCASED text;
#   2. the unit of analysis is BINARY per document, not a count. Bathina's
#      "prevalence" is the fraction of a user's tweets containing >= 1 CDS.
#      See the warning under UNIT OF ANALYSIS below.

_COMPILED = tuple(
    (cat, marker, tuple(re.compile(r"\b{}\b".format(re.escape(s.lower())))
                        for s in (marker,) + variants))
    for cat, marker, variants in CDS
)


def contains_cds(text: str, marker: str) -> bool:
    """True iff `text` contains `marker` or one of its variants (Bathina rule)."""
    low = text.lower()
    for cat, m, pats in _COMPILED:
        if m == marker:
            return any(p.search(low) for p in pats)
    raise KeyError(marker)


def annotate(text: str) -> dict:
    """Annotate one document (Bathina's 'tweet' unit).

    Returns {"markers": [...], "categories": [...], "any": bool}.
    `markers` lists every matched n-gram; `categories` the distinct categories.
    Binary per document, exactly as in the paper.
    """
    low = text.lower()
    hits, cats = [], []
    for cat, marker, pats in _COMPILED:
        if any(p.search(low) for p in pats):
            hits.append(marker)
            if cat not in cats:
                cats.append(cat)
    return {"markers": hits, "categories": cats, "any": bool(hits)}


def prevalence(documents, per_category: bool = False):
    """Bathina's prevalence: fraction of documents containing >= 1 CDS.

    Pass the documents of ONE subject (one agent, one run). With
    per_category=True returns {category: fraction}. This is the quantity the
    paper compares between cohorts; do not substitute a per-token rate.
    """
    docs = list(documents)
    n = len(docs)
    if n == 0:
        return {c: 0.0 for c in CATEGORIES} if per_category else 0.0
    anns = [annotate(d) for d in docs]
    if not per_category:
        return sum(a["any"] for a in anns) / n
    return {c: sum(c in a["categories"] for a in anns) / n for c in CATEGORIES}


# First-person-pronoun subset, for the robustness check the paper itself runs
# (Table 3, PR1: results recomputed after dropping every CDS containing "I",
# "me", "my", "mine", "myself"). This matters MORE for us than for them: a
# reasoning trace is first-person by construction, far more so than a tweet.
_FPP = re.compile(r"\b(i|me|my|mine|myself)\b", re.IGNORECASE)
CDS_WITHOUT_FPP: tuple[tuple[str, str, tuple[str, ...]], ...] = tuple(
    e for e in CDS if not _FPP.search(e[1])
)


# ===========================================================================
# APPLICABILITY TO SLOT-MACHINE REASONING TRACES
# ===========================================================================
#
# WHAT THE TWO TAXONOMIES ARE
# ---------------------------
# Bathina's 12 categories are Beck's clinical CBT distortions: content-general
# reasoning errors organised around the depressive cognitive triad (negative
# view of self, world, future). Gambling cognitive distortions (Ladouceur &
# Walker's think-aloud tradition; the GRCS/GBQ/IBS/GCI instruments reviewed in
# Goodie & Fortune 2013, which our convergent_codebook.py operationalises) are
# domain-specific errors about randomness, probability and agency: illusion of
# control, gambler's fallacy, hot hand, near-miss, superstition, chasing.
#
# They are NOT nested and NOT the same construct. The relation is structural
# analogy, not lexical overlap:
#
#   illusion of control  ~  Personalizing + Magnification   (agency over outcome)
#   gambler's fallacy    ~  Overgeneralizing                (streak -> law)
#   chasing              ~  Should statements + Mental filtering ("if I just")
#   self-serving bias    ~  Personalizing, with the valence REVERSED
#
# THE POLARITY PROBLEM — the single most important caveat
# ------------------------------------------------------
# Bathina's lexicon is polarity-locked to depressive pessimism. All 8
# Fortune-telling n-grams are negated futures ("I will not", "it will not").
# All 11 Catastrophizing n-grams predict failure ("will fail", "will go
# wrong"). Personalizing is self-BLAME ("my fault", "blame me", "I caused").
# Gambling cognition is polarity-locked the other way: optimistic prediction
# ("it's due", "this one hits"), self-CREDIT for wins, minimisation of loss.
# Consequence: an agent escalating into disordered betting can show *falling*
# Bathina Catastrophizing and Fortune-telling. A straight application does not
# merely lose power, it can point the wrong way.
#
# THE DEAD-WEIGHT PROBLEM
# -----------------------
# Mindreading (72 n-grams) + Labeling and mislabeling (44) = 116/241 = 48% of
# the lexicon. Both are unusable here for the same two reasons:
#   * Construct: a solo slot-machine trace has no other mind to read; there is
#     no second party whose beliefs are being inferred.
#   * Lexicon: they are the most promiscuous strings in the list. Mindreading
#     contains "you know", "we know", "you think", "they know" — discourse
#     filler in any deliberative text. Labeling contains "it is a", "that is
#     a", "a huge", "a total" — bare copular frames that fire on "it is a 5%
#     chance" or "that is a total of 40 credits". These will dominate any
#     unrestricted density measure with pure noise.
# Corroboration from the source paper's own Table 4: these two categories are
# also its WEAKEST discriminators (only 9.7% and 34.1% of their n-grams reached
# significance), while Personalizing / Emotional reasoning / Overgeneralizing
# reached 57.1%, and the three highest individual prevalence ratios in the
# whole study were "if it only", "because my" and "because I feel" — Mental
# filtering, Personalizing, Emotional reasoning. The categories that survive
# on construct grounds are the same ones that carried their result.
#
# UNIT OF ANALYSIS — a confound specific to us
# --------------------------------------------
# Bathina scores BINARY per tweet and averages within user. Reasoning-trace
# length varies systematically with condition (an escalating agent writes more,
# and reasons longer before betting). A per-trace COUNT of CDS is therefore
# confounded with verbosity. Segment each trace into sentences, score binary
# per sentence, and take the within-subject fraction, i.e. use `prevalence`
# above. Report trace length as a covariate regardless.
#
# PER-CATEGORY VERDICT
# --------------------
APPLICABILITY: dict[str, dict] = {
    "Personalizing": {
        "verdict": "usable, valence-inverted",
        "n": 14,
        "why": (
            "Nearest lexical proxy for illusion of control and self-serving "
            "attribution ('because I', 'I caused', 'I am responsible'). But "
            "Bathina's sense is self-blame for bad outcomes; gambling's "
            "self-serving bias is self-credit for wins. Same strings, opposite "
            "clinical meaning. Must be read jointly with outcome valence."
        ),
    },
    "Emotional Reasoning": {
        "verdict": "usable, high construct fit",
        "n": 7,
        "why": (
            "'because I feel', 'it feels', 'still feels' is exactly the form of "
            "'I feel lucky' / 'it feels due'. Highest construct fit of any "
            "category, but only 7 n-grams, so low sensitivity."
        ),
    },
    "Mental Filtering": {
        "verdict": "usable, high construct fit",
        "n": 14,
        "why": (
            "'if I just', 'if I only', 'if it just', 'if it only' is the "
            "canonical chasing frame ('if I just play one more'). Carried the "
            "single highest prevalence ratio in the source paper."
        ),
    },
    "Overgeneralizing": {
        "verdict": "usable, partial",
        "n": 21,
        "why": (
            "'always happens', 'happens every time', 'all the time' is the "
            "closest lexical trace of streak reasoning. Does NOT encode the "
            "independence violation that defines the gambler's fallacy — it "
            "catches the surface form, not the inference."
        ),
    },
    "Should statements": {
        "verdict": "usable but noisy",
        "n": 5,
        "why": (
            "'must', 'have to', 'has to' maps onto impaired control ('I have "
            "to keep going'). But in a decision trace 'I should bet 10' is a "
            "deliberative modal, not a distortion. Expect a high false-positive "
            "rate; consider requiring a first-person continuation."
        ),
    },
    "Magnification and Minimization": {
        "verdict": "usable, partial",
        "n": 8,
        "why": (
            "'not matter', 'not count', 'the only thing' fits loss minimisation "
            "and stake magnification. 'best'/'worst' are bare superlatives that "
            "fire on ordinary strategy talk."
        ),
    },
    "Dichotomous Reasoning": {
        "verdict": "marginal",
        "n": 23,
        "why": (
            "'always', 'never', 'all', 'nothing', 'only', 'every' are "
            "high-frequency quantifiers. In a probability-reasoning trace they "
            "appear as correct quantification ('never independent', 'all "
            "outcomes'), not distortion. Low specificity."
        ),
    },
    "Disqualifying the Positive": {
        "verdict": "marginal, but one real signal",
        "n": 14,
        "why": (
            "Mostly generic concessives. The 'X but' frame does however catch "
            "post-win discounting ('I won but...'), which is a genuine chasing "
            "marker."
        ),
    },
    "Catastrophizing": {
        "verdict": "polarity-mismatched",
        "n": 11,
        "why": (
            "All 11 predict failure. Gambling distortion predicts success. May "
            "move DOWN as gambling distortion goes UP. Retain only as a "
            "direction-of-effect check, never as part of a pooled score."
        ),
    },
    "Fortune-telling": {
        "verdict": "polarity-mismatched",
        "n": 8,
        "why": (
            "All 8 are negated futures ('I will not', 'it will not'). The "
            "gambling analogue is the affirmative future, which this lexicon "
            "does not contain at all. Same caveat as Catastrophizing."
        ),
    },
    "Labeling and mislabeling": {
        "verdict": "NOT applicable",
        "n": 44,
        "why": (
            "'I am a', 'it is a', 'that is a', 'a huge', 'a total' are bare "
            "copular/determiner frames that fire on ordinary numeric prose "
            "('that is a total of 40 credits'). 18% of the lexicon, near-zero "
            "construct validity here. Exclude."
        ),
    },
    "Mindreading": {
        "verdict": "NOT applicable",
        "n": 72,
        "why": (
            "No other mind in a solo slot-machine trace. Simultaneously the "
            "most promiscuous strings in the lexicon ('you know', 'we know', "
            "'you think'). 30% of the lexicon, pure noise for us. Exclude."
        ),
    },
}

# The five categories with a priori construct overlap with gambling cognition.
# FREEZE THIS BEFORE LOOKING AT ANY OUTCOME. Chosen on the construct grounds
# argued above, not on our data. Report the full 241 alongside it.
RESTRICTED_CATEGORIES: tuple[str, ...] = (
    "Personalizing",
    "Emotional Reasoning",
    "Mental Filtering",
    "Overgeneralizing",
    "Should statements",
)
CDS_RESTRICTED = tuple(e for e in CDS if e[0] in RESTRICTED_CATEGORIES)

# Categories to exclude outright on construct grounds (48% of the lexicon).
EXCLUDED_CATEGORIES: tuple[str, ...] = ("Mindreading", "Labeling and mislabeling")

# Categories whose sign is expected to INVERT relative to gambling distortion.
POLARITY_INVERTED_CATEGORIES: tuple[str, ...] = ("Catastrophizing", "Fortune-telling")


# ===========================================================================
# HOW WE PROPOSE TO USE IT
# ===========================================================================
#
# (a) Bathina-241 as the primary distortion measure.
#     +  Published, expert-panel-validated (10 CBT clinicians), zero researcher
#        degrees of freedom, directly comparable to a real literature (Nature
#        Hum Behav 2021; PNAS 2021 historical corpora; Commun Psychol 2025
#        polarization), and the matching code is the authors' own.
#     -  48% of the lexicon is construct-irrelevant and lexically promiscuous
#        here; the measured construct is depression, not gambling; polarity is
#        inverted for two categories; a null is uninterpretable, because absence
#        of depressive distortion says nothing about gambling distortion. The
#        authors themselves warn these phrases are normal English and must not
#        be read as diagnosis.
#     Verdict: not defensible as the primary measure.
#
# (b) Gambling codebook primary, Bathina-241 as a pre-registered DISCRIMINANT
#     control.  <-- RECOMMENDED
#     The polarity/construct mismatch becomes the point rather than the flaw.
#     If our gambling-specific distortion rate rises with betting escalation
#     while general Bathina CDS prevalence stays flat, that is discriminant
#     validity: the effect is gambling cognition, not "the agent got more
#     emotional / more absolutist / more verbose". If both rise together,
#     suspect a generic affect-or-verbosity confound and say so.
#     +  Turns a published, independently validated instrument into a genuine
#        negative control; costs nothing; interpretable in both directions.
#     -  Requires our gambling codebook to stand on its own (it is our own
#        deduction from Goodie & Fortune 2013, see convergent_codebook.py, not
#        a published lexicon) — provenance asymmetry must be stated plainly.
#
# (c) Restricted-Bathina (the 5 categories in RESTRICTED_CATEGORIES) as a
#     secondary, pre-registered convergent measure.
#     +  Drops the 48% dead weight; the retained five are precisely the ones
#        that carried the original paper's effect (57.1% of their n-grams
#        significant, and all three top-PR n-grams).
#     -  The subset is chosen by us. It is only credible if frozen before any
#        outcome is computed and reported next to the full 241.
#
# (d) Category profile instead of a single density: report all 12 category
#     prevalences per condition with Bathina's own bootstrap prevalence-ratio
#     machinery (resample subjects with replacement, 95% CI on P_treat/P_ctrl,
#     CI excluding 1.0 = significant).
#     +  Statistically identical to the source paper, so the comparison to
#        their published PRs is like-for-like; shows the profile shape, which
#        is more informative than one number.
#     -  12 simultaneous intervals; needs multiplicity control.
#
# PLAN: (b) as the design, with (c) and (d) as the reporting form. Primary =
# gambling codebook. Control = full Bathina-241 prevalence. Secondary =
# restricted-5 prevalence. Presentation = 12-category profile with bootstrap
# PRs. Always also report: trace length, and the CDS_WITHOUT_FPP re-run.
