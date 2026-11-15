"""
Residual (rule-guided) G2P model: the NN+HR hybrid from paper/research_plan.md Q2.

Same seq2seq architecture and training loop as the plain model (NeuralG2P.py),
but the encoder input carries the language's rule-based prediction as a hint:

  plain     (NeuralG2P.fit_model):  orthography -> gold IPA
  residual  (fit_residual):          orthography + "|" + rule prediction -> gold IPA

so the network can learn a residual correction on top of the rules instead of the
whole mapping from scratch. The hint comes from each language's own hand-written
rules (LanguageRules.rule_g2p: ports of the src/HAND_RULES scripts).

Like NeuralG2P.py, this is a module, not a script: LearningCurve.py imports it
to train residual models next to plain ones. No matplotlib import.
"""

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for LanguageRules
from NeuralG2P import SEED, EMB_DIM, HID_DIM, fit_model, levenshtein
from LanguageRules import rule_g2p

SEP = "|"

# Rule-based PER (%) as reported by the HAND_RULES scripts (their CER on FINAL_DATA);
# drawn as the rule-based baseline line on the learning-curve plots.
REPORTED_RULE_PER = {"karelian": 1.47, "livonian": 0.27, "ingrian": 1.09}


def guided_pairs(pairs, language):
    """(grapheme, phoneme) -> (grapheme + SEP + rule_prediction, phoneme), using
    that language's own rule-based G2P (LanguageRules.py)."""
    return [(f"{g}{SEP}{rule_g2p(language, g)}", p) for g, p in pairs]


def source_grapheme(src):
    """Strip the rule hint back off a residual model's 'grapheme|hint' input."""
    return src.partition(SEP)[0]


def fit_residual(language, train_pairs, dev_pairs, test_pairs=None, seed=SEED, label=None):
    """Train a residual model on plain (grapheme, phoneme) pairs: adds the rule hint
    to every split, then calls NeuralG2P.fit_model. Same return value as fit_model."""
    return fit_model(
        guided_pairs(train_pairs, language),
        guided_pairs(dev_pairs, language),
        guided_pairs(test_pairs, language) if test_pairs is not None else None,
        seed=seed, label=label,
    )


def rule_baseline_per_acc(pairs, language):
    """PER/accuracy of the rule-based pipeline alone on these pairs, using
    the same total-edits/total-length definition as evaluate_detailed() in
    NeuralG2P.py, so it's directly comparable to the neural numbers."""
    total_edits, total_len, exact = 0, 0, 0
    for grapheme, phoneme in pairs:
        phoneme = phoneme.replace("ˌ", "")  # rules have no secondary stress: strip from gold
        pred = rule_g2p(language, grapheme)
        total_edits += levenshtein(pred, phoneme)
        total_len += len(phoneme)
        exact += int(pred == phoneme)
    return total_edits / total_len, exact / len(pairs)


def save_residual_checkpoint(fit, path):
    """Save a fit_residual() result (state dict + vocab + metrics) to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": fit["state"],
        "src_stoi": fit["src_stoi"], "src_itos": fit["src_itos"],
        "tgt_stoi": fit["tgt_stoi"], "tgt_itos": fit["tgt_itos"],
        "emb_dim": EMB_DIM, "hid_dim": HID_DIM, "separator": SEP,
        "dev_per": fit["dev_per"], "dev_acc": fit["dev_acc"],
        "test_per": fit["test_per"], "test_acc": fit["test_acc"],
    }, path)
    print(f"saved {path}")
