"""
Cross-lingual G2P evaluation: zero-shot transfer grid + joint multilingual model.

Two parts, run together by default (select one with -p/--part):

transfer
    Loads each language's saved model (models/<language>_g2p_model.pt, produced by
    TrainMonolingualModels.py) and evaluates it on every language's persisted TEST split
    (data/splits/<language>_test.csv, see MakeSplits.py), for all 3x3 = 9
    (trained_on, tested_on) combinations. The 3 in-language cells (trained_on ==
    tested_on) reproduce that model's own test PER/accuracy; the 6 off-diagonal
    cells are the actual cross-lingual transfer numbers (see
    paper/research_plan.md, Q3).

    Each model is evaluated with its OWN vocabulary (src_stoi/tgt_stoi/tgt_itos
    saved in its checkpoint) -- this is a zero-shot transfer test, not
    fine-tuning. Characters in a foreign language's orthography that the model
    never saw during training fall back to <unk>, which is itself part of what's
    being measured.

    Writes outputs/cross_lingual_results.csv (aggregate PER/accuracy per
    combination) and outputs/cross_lingual_failures.csv (every mismatched
    prediction -- gold != prediction -- across all 9 combinations). Needs all
    three models/<language>_g2p_model.pt files (TrainMonolingualModels.py).

multilingual
    The in-domain counterpart to the transfer grid: trains ONE model on the pooled
    train splits of all three languages (pooled dev splits for checkpointing), then
    evaluates it on each language's held-out test split separately. Its shared
    vocab covers every character, so each test language is in-domain. Compare its
    per-language test PER against the diagonal of the transfer grid (each
    language's own monolingual model).

    Saves the model + shared vocab to models/multilingual_g2p_model.pt and a
    per-language results table to outputs/multilingual_results.csv.

Both results CSVs feed PlotCrossLingualHeatmap.py.
"""

import argparse
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from NeuralG2P import (
    ROOT, SEED, DEVICE, LANGUAGES, EMB_DIM, HID_DIM, N_EPOCHS, LEARNING_RATE,
    TEACHER_FORCING_RATIO, BATCH_SIZE, PAD, SOS, EOS,
    load_splits, build_vocab, G2PDataset, make_collate_fn,
    Encoder, Decoder, Seq2Seq, run_epoch, evaluate, evaluate_detailed,
)

RESULTS_FIELDS = ["trained_on", "tested_on", "n_test", "test_per", "test_acc"]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Part 1: zero-shot transfer grid
# --------------------------------------------------------------------------

def load_model(language):
    model_path = ROOT / "models" / f"{language}_g2p_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model for {language}: {model_path} "
            f"(run TrainMonolingualModels.py first)"
        )
    ckpt = torch.load(model_path, map_location=DEVICE)
    src_stoi, src_itos = ckpt["src_stoi"], ckpt["src_itos"]
    tgt_stoi, tgt_itos = ckpt["tgt_stoi"], ckpt["tgt_itos"]
    pad_idx = src_stoi[PAD]
    tgt_pad_idx = tgt_stoi[PAD]
    sos_idx = tgt_stoi[SOS]
    eos_idx = tgt_stoi[EOS]

    encoder = Encoder(len(src_itos), ckpt["emb_dim"], ckpt["hid_dim"], pad_idx).to(DEVICE)
    decoder = Decoder(len(tgt_itos), ckpt["emb_dim"], ckpt["hid_dim"], tgt_pad_idx).to(DEVICE)
    model = Seq2Seq(encoder, decoder, pad_idx, sos_idx, eos_idx, DEVICE).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx


def run_transfer(per_lang):
    results = []
    failures = []
    for trained_on in LANGUAGES:
        model, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx = load_model(trained_on)
        for tested_on in LANGUAGES:
            pairs = per_lang[tested_on]["test"]
            per, acc, records = evaluate_detailed(
                model, pairs, src_stoi, tgt_stoi, tgt_itos, sos_idx, eos_idx
            )
            tag = "in-language" if trained_on == tested_on else "cross-lingual"
            print(f"[{tag:13s}] trained={trained_on:10s} tested={tested_on:10s} "
                  f"n={len(pairs):4d}  test_PER={per*100:6.2f}%  test_acc={acc*100:6.2f}%")
            results.append({
                "trained_on": trained_on,
                "tested_on": tested_on,
                "n_test": len(pairs),
                "test_per": per,
                "test_acc": acc,
            })
            for rec in records:
                if rec["exact_match"]:
                    continue
                failures.append({
                    "trained_on": trained_on,
                    "tested_on": tested_on,
                    "grapheme": rec["grapheme"],
                    "gold": rec["gold"],
                    "prediction": rec["prediction"],
                    "edit_distance": rec["edit_distance"],
                })

    assert len(results) == len(LANGUAGES) ** 2, "expected 9 (trained_on, tested_on) combinations"

    out_path = ROOT / "outputs" / "cross_lingual_results.csv"
    write_csv(out_path, RESULTS_FIELDS, results)
    print(f"saved {out_path}")

    failures_path = ROOT / "outputs" / "cross_lingual_failures.csv"
    write_csv(failures_path,
              ["trained_on", "tested_on", "grapheme", "gold", "prediction", "edit_distance"],
              failures)
    print(f"saved {failures_path} ({len(failures)} failures across all 9 combinations)")


# --------------------------------------------------------------------------
# Part 2: joint multilingual model
# --------------------------------------------------------------------------

def run_multilingual(per_lang):
    train_pairs, dev_pairs = [], []
    for lang in LANGUAGES:
        train_pairs += per_lang[lang]["train"]
        dev_pairs += per_lang[lang]["dev"]
    print(f"[pooled] train {len(train_pairs)} / dev {len(dev_pairs)}")

    random.seed(SEED)
    torch.manual_seed(SEED)

    # Shared vocab over the pooled training data -> covers all three languages'
    # characters, so no <unk> fallback for any of the test languages.
    src_stoi, src_itos = build_vocab([g for g, _ in train_pairs])
    tgt_stoi, tgt_itos = build_vocab([p for _, p in train_pairs])
    pad_idx = src_stoi[PAD]
    tgt_pad_idx = tgt_stoi[PAD]
    sos_idx = tgt_stoi[SOS]
    eos_idx = tgt_stoi[EOS]
    print(f"[pooled] src vocab {len(src_itos)} / tgt vocab {len(tgt_itos)}")

    collate_fn = make_collate_fn(pad_idx, tgt_pad_idx)
    train_dl = DataLoader(G2PDataset(train_pairs, src_stoi, tgt_stoi),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(G2PDataset(dev_pairs, src_stoi, tgt_stoi),
                        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(src_itos), EMB_DIM, HID_DIM, pad_idx).to(DEVICE)
    decoder = Decoder(len(tgt_itos), EMB_DIM, HID_DIM, tgt_pad_idx).to(DEVICE)
    model = Seq2Seq(encoder, decoder, pad_idx, sos_idx, eos_idx, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    best_dev_loss = float("inf")
    best_state = None
    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(model, train_dl, optimizer, criterion, train=True,
                               teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        dev_loss = run_epoch(model, dev_dl, optimizer, criterion, train=False)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[pooled] epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"dev_loss {dev_loss:.4f}")

    model.load_state_dict(best_state)

    # Evaluate the single pooled model on each language's own test split.
    rows = []
    print(f"\n[pooled] best_dev_loss={best_dev_loss:.4f}")
    print(f"{'tested_on':<10} {'n_test':>6} {'test_PER':>9} {'test_acc':>9}")
    for lang in LANGUAGES:
        test_pairs = per_lang[lang]["test"]
        test_per, test_acc = evaluate(model, test_pairs, src_stoi, tgt_stoi,
                                      tgt_itos, sos_idx, eos_idx)
        print(f"{lang:<10} {len(test_pairs):>6} {test_per*100:>8.2f}% {test_acc*100:>8.2f}%")
        rows.append({
            "trained_on": "multilingual",
            "tested_on": lang,
            "n_test": len(test_pairs),
            "test_per": test_per,
            "test_acc": test_acc,
        })

    out_path = ROOT / "outputs" / "multilingual_results.csv"
    write_csv(out_path, RESULTS_FIELDS, rows)
    print(f"\nsaved {out_path}")

    model_path = ROOT / "models" / "multilingual_g2p_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "src_stoi": src_stoi, "src_itos": src_itos,
        "tgt_stoi": tgt_stoi, "tgt_itos": tgt_itos,
        "emb_dim": EMB_DIM, "hid_dim": HID_DIM,
        "languages": LANGUAGES,
        "results": rows,
    }, model_path)
    print(f"saved {model_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("-p", "--part", choices=["transfer", "multilingual", "both"],
                    default="both",
                    help="transfer = 3x3 zero-shot grid from the saved monolingual models; "
                         "multilingual = train + test the joint model; both (default)")
    args = ap.parse_args()

    print("device:", DEVICE)
    per_lang = {lang: load_splits(lang) for lang in LANGUAGES}
    for lang in LANGUAGES:
        s = per_lang[lang]
        print(f"[{lang}] train {len(s['train'])} / dev {len(s['dev'])} / "
              f"test {len(s['test'])}")

    if args.part in ("transfer", "both"):
        print("\n=== zero-shot transfer grid ===")
        run_transfer(per_lang)
    if args.part in ("multilingual", "both"):
        print("\n=== joint multilingual model ===")
        run_multilingual(per_lang)


if __name__ == "__main__":
    main()
