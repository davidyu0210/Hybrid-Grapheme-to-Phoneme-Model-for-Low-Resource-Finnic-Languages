"""
Learning curves: plain NN vs. residual (rule-guided) NN vs. rules alone.

For each FRACTION of a language's train pool (NeuralG2P.FRACTIONS, 10%..100%),
trains two models from scratch on the *same* sampled subset, so any gap is
attributable to the rule hint:

  plain     NeuralG2P.fit_model      orthography -> gold IPA
  residual  ResidualG2P.fit_residual orthography + "|" + rule prediction -> gold IPA

and scores both on the fixed dev split. The rule-based system's PER is drawn as a
reference line.

Subcommands:

  run   Sweep the fractions. Dev PER/acc at every fraction goes to
        outputs/<language>[_norm|_full]_plain_vs_residual_learning_curve_results.csv,
        plotted to figures/<language>[_norm|_full]_plain_vs_residual_learning_curve.png.
        The 100% fraction is the full-train-pool model, so it doubles as the final
        model: only there is the test split evaluated, and only there are these
        written (replacing just the (language, condition) rows being re-run):
          outputs/plain_vs_residual_results.csv   rule-only, plain and residual PER/acc
          outputs/plain_vs_residual_failures.csv  every non-exact-match test prediction
                                                of both models, plus the rule hint
          models/<language>[_norm|_full]_residual_g2p_model.pt
                                                residual state dict + vocab (the plain
                                                model is not saved; the full-data plain
                                                checkpoint comes from TrainMonolingualModels.py)
        --language/-l (default: all three). Plotting needs matplotlib; if it can't be
        imported (see the DYLD_LIBRARY_PATH note in CLAUDE.md) the CSVs are still
        written and `plot` can be run afterwards.

  plot  Re-plot a language's saved curve from its results CSV (no retraining).

Data conditions (-c/--condition): Livonian and Ingrian have far more data than
Karelian (690 pairs total), which would confound the cross-language comparison.
`norm` randomly subsamples (seeded) each of their train/dev/test splits to
N_TOTAL=700 examples in the same 80/10/10 proportions (560/70/70), keeping the
original train/dev/test membership; `full` uses the whole splits; `both` (default)
runs both. Karelian's splits are smaller than the targets, so it runs once.
"""

import argparse
import csv
import random
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for LanguageRules
from NeuralG2P import ROOT, SEED, DEVICE, LANGUAGES, FRACTIONS, load_splits, fit_model
from ResidualG2P import (
    REPORTED_RULE_PER, fit_residual, rule_baseline_per_acc, save_residual_checkpoint,
    source_grapheme,
)
from LanguageRules import rule_g2p

N_TOTAL = 700
SPLIT_PROPORTIONS = {"train": 0.8, "dev": 0.1, "test": 0.1}

RESULTS_FIELDS = ["language", "condition", "n_train", "n_dev", "n_test",
                  "rule_test_per", "rule_test_acc",
                  "plain_dev_per", "plain_dev_acc", "plain_test_per", "plain_test_acc",
                  "residual_dev_per", "residual_dev_acc", "residual_test_per", "residual_test_acc"]
FAILURE_FIELDS = ["language", "condition", "model", "grapheme", "rule_prediction",
                  "gold", "prediction", "edit_distance"]
CURVE_FIELDS = ["fraction", "n_train", "dev_per_plain", "dev_acc_plain",
                "dev_per_residual", "dev_acc_residual", "rule_dev_per"]


# --------------------------------------------------------------------------
# Data conditions and file names
# --------------------------------------------------------------------------

def subsample_split(pairs, split_name, language):
    """Seeded random sample of at most round(N_TOTAL * proportion) pairs."""
    target = round(N_TOTAL * SPLIT_PROPORTIONS[split_name])
    if len(pairs) <= target:
        return pairs
    rng = random.Random(f"{SEED}-{language}-{split_name}")
    return rng.sample(pairs, target)


def conditions_for(language, choice):
    """Data conditions to run for a language. Karelian's splits are smaller than
    N_TOTAL so subsampling never touches it: one condition, 'full'==normalized."""
    if language == "karelian":
        return ["full"]
    return ["norm", "full"] if choice == "both" else [choice]


def tag(language, condition):
    """File-name suffix: Karelian has a single version, others are _norm / _full."""
    return "" if language == "karelian" else f"_{condition}"


def curve_paths(language, condition):
    t = tag(language, condition)
    return (ROOT / "outputs" / f"{language}{t}_plain_vs_residual_learning_curve_results.csv",
            ROOT / "figures" / f"{language}{t}_plain_vs_residual_learning_curve.png")


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------

def failure_rows(language, condition, model_name, records):
    """Non-exact-match test predictions as CSV rows, keyed by plain orthography
    (residual records carry 'grapheme|rule_hint' as their source string)."""
    rows = []
    for rec in records:
        if rec["exact_match"]:
            continue
        grapheme = source_grapheme(rec["grapheme"])
        rows.append({
            "language": language, "condition": condition, "model": model_name, "grapheme": grapheme,
            "rule_prediction": rule_g2p(language, grapheme),
            "gold": rec["gold"], "prediction": rec["prediction"],
            "edit_distance": rec["edit_distance"],
        })
    return rows


def merge_write_csv(path, fieldnames, new_rows, keys):
    """Write new_rows, keeping other (language, condition) rows already in the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if (r["language"], r.get("condition", "")) not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + new_rows)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def run_language(language, condition):
    """Sweep FRACTIONS with a plain and a residual model. Returns (results row, failure rows)."""
    splits = load_splits(language)
    if condition == "full":
        train_pool, dev_pairs, test_pairs = (splits[n] for n in ("train", "dev", "test"))
    else:
        train_pool, dev_pairs, test_pairs = (
            subsample_split(splits[name], name, language) for name in ("train", "dev", "test")
        )
    print(f"[{language}/{condition}] train pool {len(train_pool)} / dev {len(dev_pairs)} / test {len(test_pairs)}")

    rule_dev_per, rule_dev_acc = rule_baseline_per_acc(dev_pairs, language)
    rule_test_per, rule_test_acc = rule_baseline_per_acc(test_pairs, language)
    print(f"[{language}] rule-only   dev_PER={rule_dev_per*100:.2f}% dev_acc={rule_dev_acc*100:.2f}%  "
          f"test_PER={rule_test_per*100:.2f}% test_acc={rule_test_acc*100:.2f}%")

    curve = []
    plain = residual = None
    for frac in FRACTIONS:
        n_train = max(1, round(frac * len(train_pool)))
        subset = random.Random(SEED + round(frac * 1000)).sample(train_pool, n_train)
        is_full = frac == FRACTIONS[-1]
        test = test_pairs if is_full else None

        plain = fit_model(subset, dev_pairs, test, seed=SEED,
                          label=f"[{language}] plain    frac={frac:.1f}" if is_full else None)
        residual = fit_residual(language, subset, dev_pairs, test, seed=SEED,
                                label=f"[{language}] residual frac={frac:.1f}" if is_full else None)
        print(f"[{language}] frac={frac:.1f}  n_train={n_train:4d}  "
              f"plain dev_PER={plain['dev_per']*100:6.2f}% dev_acc={plain['dev_acc']*100:6.2f}%  |  "
              f"residual dev_PER={residual['dev_per']*100:6.2f}% dev_acc={residual['dev_acc']*100:6.2f}%")
        curve.append((frac, n_train, plain["dev_per"], plain["dev_acc"],
                      residual["dev_per"], residual["dev_acc"], rule_dev_per))

    # After the loop, plain/residual hold the 100% (full-train-pool) models.
    curve_path, _ = curve_paths(language, condition)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CURVE_FIELDS)
        writer.writerows(curve)
    print(f"wrote {curve_path}")

    save_residual_checkpoint(
        residual, ROOT / "models" / f"{language}{tag(language, condition)}_residual_g2p_model.pt")

    row = {
        "language": language, "condition": condition,
        "n_train": len(train_pool), "n_dev": len(dev_pairs), "n_test": len(test_pairs),
        "rule_test_per": rule_test_per, "rule_test_acc": rule_test_acc,
        **{f"plain_{k}": plain[k] for k in ("dev_per", "dev_acc", "test_per", "test_acc")},
        **{f"residual_{k}": residual[k] for k in ("dev_per", "dev_acc", "test_per", "test_acc")},
    }
    failures = (failure_rows(language, condition, "plain", plain["test_records"])
                + failure_rows(language, condition, "residual", residual["test_records"]))
    return row, failures


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot_curve(language, condition, csv_override=None, out_override=None):
    """Plot the plain-vs-residual curve from the saved results CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_path, plot_path = curve_paths(language, condition)
    curve_path, plot_path = csv_override or curve_path, out_override or plot_path
    fracs, plain_pers, residual_pers, rule_pers = [], [], [], []
    with open(curve_path) as f:
        for row in csv.DictReader(f):
            fracs.append(float(row["fraction"]))
            plain_pers.append(float(row["dev_per_plain"]) * 100)
            residual_pers.append(float(row["dev_per_residual"]) * 100)
            rule_pers.append(float(row["rule_dev_per"]) * 100)

    plt.figure(figsize=(7, 5))
    plt.plot(plain_pers, marker="o", label="plain NN")
    plt.plot(residual_pers, marker="o", label="residual NN (rule-guided)")
    rb = REPORTED_RULE_PER.get(language, rule_pers[0])
    plt.axhline(rb, linestyle="--", color="gray", label=f"rule-based ({rb:.2f}%)")
    plt.gca().set_xticks(range(len(fracs)))
    plt.gca().set_xticklabels([f"{int(f*100)}%" for f in fracs])
    plt.xlabel("training data used (% of train pool)")
    plt.ylabel("dev phoneme error rate (%)")
    desc = " (full data)" if language != "karelian" and condition == "full" else ""
    plt.title(f"{language.capitalize()}{desc}: plain vs. residual")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"saved {plot_path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="train plain + residual across fractions; write curve, "
                                       "results, failures, checkpoint")
    p_run.add_argument("-l", "--language", choices=LANGUAGES, default=None,
                       help="Language (default: all three)")
    p_run.add_argument("-c", "--condition", choices=["norm", "full", "both"], default="both",
                       help="Livonian/Ingrian data: normalized to 700, full, or both (default). "
                            "Karelian always runs once.")
    p_plot = sub.add_parser("plot", help="re-plot a saved curve (no retraining)")
    p_plot.add_argument("-l", "--language", choices=LANGUAGES, default="karelian",
                        help="Language (default: karelian)")
    p_plot.add_argument("-c", "--condition", choices=["norm", "full"], default="full",
                        help="Data condition (ignored for karelian)")
    p_plot.add_argument("--csv", type=Path, default=None, help="read this results CSV instead of outputs/")
    p_plot.add_argument("--out", type=Path, default=None, help="save the figure here instead of figures/")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.command == "plot":
        plot_curve(args.language, args.condition, args.csv, args.out)
        return

    print("device:", DEVICE)
    languages = [args.language] if args.language else LANGUAGES
    rows, failures, keys, runs = [], [], set(), []
    for language in languages:
        for condition in conditions_for(language, args.condition):
            row, language_failures = run_language(language, condition)
            rows.append(row)
            failures.extend(language_failures)
            keys.add((language, condition))
            runs.append((language, condition))

    merge_write_csv(ROOT / "outputs" / "plain_vs_residual_results.csv", RESULTS_FIELDS, rows, keys)
    merge_write_csv(ROOT / "outputs" / "plain_vs_residual_failures.csv", FAILURE_FIELDS, failures, keys)

    for language, condition in runs:
        try:
            plot_curve(language, condition)
        except ImportError as e:
            print(f"[{language}/{condition}] could not plot ({e}); CSVs are saved, run `plot` after fixing matplotlib")


if __name__ == "__main__":
    main()
