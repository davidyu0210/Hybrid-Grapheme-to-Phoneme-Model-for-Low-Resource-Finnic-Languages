"""
Generate the fixed 80/10/10 train/dev/test split for each language from the
final datasets in data/FINAL_DATA/ (lowercased, no secondary stress, no syllable dots)
and write it to data/splits/<language>_{train,dev,test}.csv.

The Livonian file spells orthographic g with the IPA letter U+0261; it is
converted back to ASCII g in the orthography so the rule-based G2P recognizes it.

Every other script (LearningCurve.py, TrainMonolingualModels.py,
CrossLingualEval.py) reads these files directly instead of re-deriving the
split by reshuffling data/<language>_terms.csv on every run. Re-run this only
when a language's raw *_terms.csv changes -- not as part of a normal training
run.
"""

import csv
import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "NeuralNetwork"))  # NeuralG2P & co. live there
from NeuralG2P import ROOT, SEED, LANGUAGES, load_pairs, split_paths


FINAL_FILES = {"karelian": "krl-final.csv", "livonian": "liv-final.csv", "ingrian": "izh-final.csv"}


def split_pairs(pairs, seed=SEED):
    shuffled = pairs[:]
    random.seed(seed)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_dev = max(1, round(0.1 * n))
    n_test = max(1, round(0.1 * n))
    dev_pairs = shuffled[:n_dev]
    test_pairs = shuffled[n_dev:n_dev + n_test]
    train_pairs = shuffled[n_dev + n_test:]
    return train_pairs, dev_pairs, test_pairs


def write_pairs(path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for grapheme, phoneme in pairs:
            writer.writerow([grapheme, f"/{phoneme}/"])


def main():
    for language in LANGUAGES:
        data_path = ROOT / "data" / "FINAL_DATA" / FINAL_FILES[language]
        pairs = [(g.replace("\u0261", "g"), p) for g, p in load_pairs(data_path)]
        pairs = list(dict.fromkeys(pairs))  # the g fix can create duplicates
        train_pairs, dev_pairs, test_pairs = split_pairs(pairs)

        paths = split_paths(language)
        write_pairs(paths["train"], train_pairs)
        write_pairs(paths["dev"], dev_pairs)
        write_pairs(paths["test"], test_pairs)

        print(f"[{language}] {len(pairs)} pairs -> train {len(train_pairs)} / "
              f"dev {len(dev_pairs)} / test {len(test_pairs)}  -> {paths['train'].parent}")


if __name__ == "__main__":
    main()
