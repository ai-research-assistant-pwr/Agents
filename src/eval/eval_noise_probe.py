"""
eval_noise_probe.py — Generator Listening Probe: Analiza Statystyczna
======================================================================

Czy generator faktycznie „słucha" komunikatu retrievera?

Skrypt wczytuje trajektorie zebrane przez run_eval_noisy.sh / run_eval_clean.sh:
  • workflow_logs/<probe>/eval_noisy/eval/  — przebiegi z noise_probability=1.0
  • workflow_logs/<probe>/eval_clean/eval/  — przebiegi z noise_probability=0.0

Uwaga: scientific_workflow.py zapisuje trajektorie eval do podkatalogu /eval/
wewnątrz debug_dir. Skrypt automatycznie go wykrywa i wchodzi poziom niżej.

Ponieważ oba joby używają n_samples_per_prompt=10 na tym samym zbiorze eval,
każdy prompt jest reprezentowany przez 10 trajektorii w każdym warunku.
Skrypt paruje trajektorie po prompcie (wszystkie N próbek clean z wszystkimi
N próbkami noisy dla tego samego promptu → N² par per prompt dla H3,
lub agregacja per prompt → jedna wartość per prompt dla H1/H2).

Testowane hipotezy
------------------
  H1 — NAGRODA: Czy nagroda końcowa różni się między warunkami?
       Test: Mann-Whitney U (jednostronny: clean > noisy)
       Poziom obserwacji: trajektoria (każda próbka osobno)

  H2 — COS-SIM DO GOLD-LABEL: Czy hipotezy clean są bliżej ground truth?
       Test: Mann-Whitney U (jednostronny: clean > noisy)
       Poziom obserwacji: trajektoria

  H3 — DYWERGENCJA WEWNĄTRZ PROMPTU: Czy hipotezy clean i noisy różnią się
       semantycznie dla tego samego pytania?
       Metryka: średnia(1 − cos(emb_clean_hyp_i, emb_noisy_hyp_j)) per prompt
       Test: Wilcoxon signed-rank (parowany po prompcie, jednostronny: div > 0)

Output
------
  <output_dir>/
    ├── noise_probe_report.txt        — pełny raport tekstowy
    ├── noise_probe_summary.json      — wyniki testów w JSON
    ├── noise_probe_rewards.png       — boxplot nagród clean vs noisy
    ├── noise_probe_similarity.png    — boxplot cos-sim do gold-label
    └── noise_probe_divergence.png    — boxplot dywergencji per prompt

Użycie
------
  python src/eval/eval_noise_probe.py \\
      --noisy_dir  Agents/workflow_logs/noise_probe/eval_noisy \\
      --clean_dir  Agents/workflow_logs/noise_probe/eval_clean \\
      --output_dir Agents/eval_results/noise_probe \\
      [--embed_host localhost] [--embed_port 8000] \\
      [--embed_model Qwen/Qwen3-Embedding-4B]

  Możesz też podać ścieżki bezpośrednio do podkatalogu /eval/ — skrypt
  wykryje to automatycznie i nie wejdzie podwójnie.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


# =============================================================================
# Embedding helpers
# =============================================================================

def _embed_via_vllm(
    texts: List[str], host: str, port: int, model: str
) -> Optional[np.ndarray]:
    """Zwraca macierz embeddingów (n, d) z lokalnego serwera vLLM lub None przy błędzie."""
    try:
        import requests

        url = f"http://{host}:{port}/v1/embeddings"
        payload = {"model": model, "input": texts}
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        embs = [item["embedding"] for item in data["data"]]
        return np.array(embs, dtype=np.float32)
    except Exception as exc:
        print(f"[WARN] vLLM embedding failed: {exc}")
        return None


def _embed_via_sentence_transformers(texts: List[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    except ImportError:
        raise RuntimeError(
            "sentence-transformers not installed and vLLM server unreachable.\n"
            "Zainstaluj: pip install sentence-transformers"
        )


def get_embeddings(
    texts: List[str],
    embed_host: str,
    embed_port: int,
    embed_model: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Pobiera embeddingi, preferując vLLM; fallback → sentence-transformers."""
    all_embs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = _embed_via_vllm(batch, embed_host, embed_port, embed_model)
        if embs is None:
            print("[INFO] Fallback: używam sentence-transformers (all-MiniLM-L6-v2).")
            return _embed_via_sentence_transformers(texts)
        all_embs.append(embs)

    return np.vstack(all_embs)


def cosine_similarity_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity; obie macierze (n, d)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.einsum("ij,ij->i", a_n, b_n)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pełna macierz podobieństwa (n_a, n_b)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_n @ b_n.T


# =============================================================================
# Parsowanie trajektorii
# =============================================================================

def _resolve_eval_subdir(directory: str) -> str:
    """
    scientific_workflow.py zapisuje trajektorie eval do <debug_dir>/eval/.
    Jeśli katalog podany przez użytkownika nie zawiera traj_*.json ale ma
    podkatalog eval/, wchodzimy do niego automatycznie.
    """
    flat = glob.glob(os.path.join(directory, "traj_*.json"))
    if flat:
        return directory  # użytkownik podał już właściwy katalog

    eval_sub = os.path.join(directory, "eval")
    if os.path.isdir(eval_sub):
        sub_files = glob.glob(os.path.join(eval_sub, "traj_*.json"))
        if sub_files:
            print(f"  Auto-wykryto podkatalog /eval/ → {eval_sub}")
            return eval_sub

    # Brak trajektorii gdziekolwiek — zwróć oryginalny, błąd wyjdzie przy load
    return directory


def _extract_hypothesis_text(turns: List[Dict]) -> str:
    """Wyciąga tekst hipotezy z ostatniego kroku generator_generate."""
    for turn in reversed(turns):
        if turn.get("turn_type") == "generator_generate":
            hyps = turn.get("hypotheses", [])
            if isinstance(hyps, list) and hyps:
                return " ".join(str(h) for h in hyps)
            return turn.get("output", "")
    return ""


def load_trajectories(directory: str) -> List[Dict]:
    """
    Wczytuje wszystkie traj_*.json z katalogu (rekurencyjnie).
    Zwraca listę rekordów z kluczami: prompt, label, final_reward, hypothesis_text.
    """
    resolved = _resolve_eval_subdir(directory)
    files = glob.glob(os.path.join(resolved, "traj_*.json"))
    if not files:
        # ostatnia szansa: pełna rekurencja
        files = glob.glob(os.path.join(resolved, "**", "traj_*.json"), recursive=True)

    records = []
    for fp in sorted(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            records.append(
                {
                    "file": fp,
                    "prompt": data.get("prompt", ""),
                    "label": data.get("label", ""),
                    "final_reward": float(data.get("total_reward", 0.0)),
                    "hypothesis_text": _extract_hypothesis_text(data.get("turns", [])),
                }
            )
        except Exception as exc:
            print(f"  [WARN] Nie można wczytać {fp}: {exc}")
    return records


def group_by_prompt(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Grupuje rekordy po treści promptu."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        grouped[r["prompt"]].append(r)
    return dict(grouped)


# =============================================================================
# Testy statystyczne
# =============================================================================

def mwu_report(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    metric_name: str,
    alternative: str = "greater",
) -> Dict:
    """Mann-Whitney U; alternative='greater' → H: median(a) > median(b)."""
    stat, p = mannwhitneyu(a, b, alternative=alternative)
    # rank-biserial effect size
    effect_r = (2 * stat / (len(a) * len(b))) - 1
    return {
        "metric": metric_name,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "U_statistic": float(stat),
        "p_value": float(p),
        "effect_r_biserial": float(effect_r),
        "alternative": f"{label_a} > {label_b}",
        "significant_at_005": bool(p < 0.05),
        "significant_at_001": bool(p < 0.01),
    }


def wilcoxon_report(
    differences: np.ndarray,
    metric_name: str,
) -> Dict:
    """
    Wilcoxon signed-rank test (jednostronny: różnice > 0).
    `differences` to wartości (clean - noisy) lub (divergence per prompt).
    """
    # scipy.stats.wilcoxon nie obsługuje alternative='greater' w starszych wersjach;
    # używamy alternative='greater' jeśli dostępne, inaczej liczymy ręcznie
    try:
        stat, p = wilcoxon(differences, alternative="greater")
    except TypeError:
        # fallback dla starszego scipy
        stat, p_two = wilcoxon(differences)
        p = p_two / 2 if np.mean(differences) > 0 else 1.0 - p_two / 2

    return {
        "metric": metric_name,
        "n_pairs": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "std_difference": float(np.std(differences)),
        "W_statistic": float(stat),
        "p_value_one_sided": float(p),
        "alternative": "divergence > 0 (clean ≠ noisy)",
        "significant_at_005": bool(p < 0.05),
        "significant_at_001": bool(p < 0.01),
    }


# =============================================================================
# Wykresy
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

_COLOR_CLEAN = "#4C72B0"
_COLOR_NOISY = "#DD8452"
_COLOR_REF   = "#aaaaaa"


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _boxplot_two(
    data_clean: np.ndarray,
    data_noisy: np.ndarray,
    title: str,
    ylabel: str,
    path: str,
    p_value: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bp = ax.boxplot(
        [data_clean, data_noisy],
        labels=["Clean\n(noise=0.0)", "Noisy\n(noise=1.0)"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], [_COLOR_CLEAN, _COLOR_NOISY]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", lw=0.4, ls=":", color=_COLOR_REF)

    if p_value is not None:
        stars = _sig_stars(p_value)
        y_max = max(
            np.percentile(data_clean, 95) if len(data_clean) else 0,
            np.percentile(data_noisy, 95) if len(data_noisy) else 0,
        )
        y_annot = y_max * 1.05
        ax.annotate(
            stars,
            xy=(1.5, y_annot),
            ha="center",
            fontsize=14,
            color="black",
        )
        ax.annotate(
            f"p={p_value:.3f}",
            xy=(1.5, y_annot),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#444444",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {path}")


def _boxplot_divergence(
    per_prompt_divergences: np.ndarray,
    title: str,
    path: str,
    p_value: Optional[float] = None,
) -> None:
    """Boxplot dywergencji per prompt + linia referencyjna zero."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bp = ax.boxplot(
        per_prompt_divergences,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    bp["boxes"][0].set_facecolor("#8172B2")
    bp["boxes"][0].set_alpha(0.75)

    ax.axhline(0, color="red", lw=1.2, ls="--", label="zero (brak dywergencji)")
    ax.set_xticklabels(["Per-prompt\ndywergencja"])
    ax.set_ylabel("Średnia odległość cosinusowa\n(1 − cos(clean, noisy))")
    ax.set_title(title, pad=8)
    ax.grid(axis="y", lw=0.4, ls=":", color=_COLOR_REF)
    ax.legend(frameon=False, fontsize=8)

    if p_value is not None:
        stars = _sig_stars(p_value)
        ax.annotate(
            f"{stars}  p={p_value:.3f}",
            xy=(1, float(np.median(per_prompt_divergences))),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {path}")


# =============================================================================
# Główna funkcja
# =============================================================================

def run_noise_probe_analysis(
    noisy_dir: str,
    clean_dir: str,
    output_dir: str,
    embed_host: str = "localhost",
    embed_port: int = 8000,
    embed_model: str = "Qwen/Qwen3-Embedding-4B",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Wczytaj trajektorie ────────────────────────────────────────────────
    print("\n[1/5] Wczytywanie trajektorii...")
    noisy_recs = load_trajectories(noisy_dir)
    clean_recs = load_trajectories(clean_dir)

    if not noisy_recs:
        sys.exit(f"ERROR: Brak trajektorii w {noisy_dir} (ani /eval/ podkatalogu)")
    if not clean_recs:
        sys.exit(f"ERROR: Brak trajektorii w {clean_dir} (ani /eval/ podkatalogu)")

    print(f"  Noisy: {len(noisy_recs)} trajektorii")
    print(f"  Clean: {len(clean_recs)} trajektorii")

    noisy_by_prompt = group_by_prompt(noisy_recs)
    clean_by_prompt = group_by_prompt(clean_recs)
    shared_prompts = sorted(set(noisy_by_prompt) & set(clean_by_prompt))

    print(f"  Promptów wspólnych: {len(shared_prompts)} "
          f"(noisy-only: {len(noisy_by_prompt)-len(shared_prompts)}, "
          f"clean-only: {len(clean_by_prompt)-len(shared_prompts)})")

    if not shared_prompts:
        sys.exit("ERROR: Brak wspólnych promptów — upewnij się że oba joby używały tego samego EVAL_DATA_PATH.")

    # ── 2. H1: Nagroda ────────────────────────────────────────────────────────
    print("\n[2/5] H1 — Nagroda końcowa (Mann-Whitney U)...")
    rewards_noisy = np.array([r["final_reward"] for r in noisy_recs])
    rewards_clean = np.array([r["final_reward"] for r in clean_recs])

    h1 = mwu_report(
        rewards_clean, rewards_noisy,
        label_a="clean", label_b="noisy",
        metric_name="final_reward",
        alternative="greater",
    )
    _print_result("H1 — Nagroda (clean > noisy)", h1)

    # ── 3. Embeddingi ─────────────────────────────────────────────────────────
    print("\n[3/5] Obliczanie embeddingów hipotez i gold-labels...")

    # Zbieramy wszystkie niepuste hipotezy do jednej paczki → jedno wywołanie serwera
    noisy_valid = [r for r in noisy_recs if r["hypothesis_text"].strip()]
    clean_valid = [r for r in clean_recs if r["hypothesis_text"].strip()]

    if not noisy_valid or not clean_valid:
        sys.exit("ERROR: Zbyt mało niepustych hipotez. Sprawdź trajektorie.")

    all_texts = (
        [r["hypothesis_text"] for r in noisy_valid]
        + [r["hypothesis_text"] for r in clean_valid]
        + [r["label"] for r in noisy_valid]
        + [r["label"] for r in clean_valid]
    )

    embs_all = get_embeddings(all_texts, embed_host, embed_port, embed_model)

    n_nv = len(noisy_valid)
    n_cv = len(clean_valid)

    embs_noisy_hyp   = embs_all[:n_nv]
    embs_clean_hyp   = embs_all[n_nv : n_nv + n_cv]
    embs_noisy_label = embs_all[n_nv + n_cv : n_nv + n_cv + n_nv]
    embs_clean_label = embs_all[n_nv + n_cv + n_nv :]

    # ── 4. H2: Cos-sim do gold-label ──────────────────────────────────────────
    print("\n[4/5] H2 — Cos-sim hipotez do gold-label (Mann-Whitney U)...")
    sim_noisy = cosine_similarity_rowwise(embs_noisy_hyp, embs_noisy_label)
    sim_clean = cosine_similarity_rowwise(embs_clean_hyp, embs_clean_label)

    h2 = mwu_report(
        sim_clean, sim_noisy,
        label_a="clean", label_b="noisy",
        metric_name="cosine_similarity_to_gold",
        alternative="greater",
    )
    _print_result("H2 — Cos-sim do gold-label (clean > noisy)", h2)

    # ── 5. H3: Dywergencja per prompt ─────────────────────────────────────────
    print("\n[5/5] H3 — Dywergencja hipotez per prompt (Wilcoxon signed-rank)...")

    # Dla każdego wspólnego promptu: średnia odległość cosinus między wszystkimi
    # parami (clean_i, noisy_j) → jedna liczba per prompt → test parowany
    per_prompt_divergences: List[float] = []

    # Zbuduj słowniki: prompt → lista embeddingów hipotez
    def _build_emb_index(
        valid_records: List[Dict], embs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        idx: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(valid_records):
            idx[r["prompt"]].append(i)
        return {
            prompt: embs[rows]
            for prompt, rows in idx.items()
            if rows
        }

    clean_emb_by_prompt = _build_emb_index(clean_valid, embs_clean_hyp)
    noisy_emb_by_prompt = _build_emb_index(noisy_valid, embs_noisy_hyp)

    for prompt in shared_prompts:
        if prompt not in clean_emb_by_prompt or prompt not in noisy_emb_by_prompt:
            continue
        ec = clean_emb_by_prompt[prompt]   # (n_clean, d)
        en = noisy_emb_by_prompt[prompt]   # (n_noisy, d)

        # Macierz podobieństwa (n_clean, n_noisy) → odległości
        sim_matrix = cosine_similarity_matrix(ec, en)
        dist_matrix = 1.0 - sim_matrix
        per_prompt_divergences.append(float(np.mean(dist_matrix)))

    per_prompt_divergences_arr = np.array(per_prompt_divergences)

    if len(per_prompt_divergences_arr) < 2:
        print("[WARN] Za mało promptów do testu Wilcoxona — pomijam H3.")
        h3: Dict = {"error": "za mało par", "n_pairs": len(per_prompt_divergences_arr)}
    else:
        h3 = wilcoxon_report(per_prompt_divergences_arr, metric_name="hypothesis_divergence_per_prompt")
        _print_result("H3 — Dywergencja hipotez clean vs noisy (per prompt)", h3)

    # ── Zapis ─────────────────────────────────────────────────────────────────
    summary = {
        "n_noisy_trajectories": len(noisy_recs),
        "n_clean_trajectories": len(clean_recs),
        "n_shared_prompts": len(shared_prompts),
        "noisy_dir": noisy_dir,
        "clean_dir": clean_dir,
        "embed_model": embed_model,
        "H1_reward": h1,
        "H2_cosine_similarity_to_gold": h2,
        "H3_hypothesis_divergence_per_prompt": h3,
    }

    json_path = os.path.join(output_dir, "noise_probe_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Wyniki JSON: {json_path}")

    _write_text_report(summary, output_dir)

    # ── Wykresy ───────────────────────────────────────────────────────────────
    _boxplot_two(
        rewards_clean, rewards_noisy,
        title="H1 — Nagroda końcowa: clean vs noisy\n(Probe: czy generator słucha retrievera?)",
        ylabel="Final reward",
        path=os.path.join(output_dir, "noise_probe_rewards.png"),
        p_value=h1["p_value"],
    )

    _boxplot_two(
        sim_clean, sim_noisy,
        title="H2 — Cos-sim hipotez do gold-label: clean vs noisy",
        ylabel="Cosine similarity",
        path=os.path.join(output_dir, "noise_probe_similarity.png"),
        p_value=h2["p_value"],
    )

    if isinstance(h3, dict) and "mean_difference" in h3 and len(per_prompt_divergences_arr) >= 2:
        _boxplot_divergence(
            per_prompt_divergences_arr,
            title="H3 — Dywergencja hipotez per prompt\n(clean ≠ noisy → generator reaguje na kanał)",
            path=os.path.join(output_dir, "noise_probe_divergence.png"),
            p_value=h3.get("p_value_one_sided"),
        )

    print(f"\n[✓] Analiza zakończona. Wyniki w: {output_dir}")


# =============================================================================
# Pomocnicze
# =============================================================================

def _print_result(heading: str, result: Dict) -> None:
    print(f"\n  {'─' * 62}")
    print(f"  {heading}")
    print(f"  {'─' * 62}")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:42s}: {v:.4f}")
        else:
            print(f"  {k:42s}: {v}")


def _write_text_report(summary: Dict, output_dir: str) -> None:
    lines = [
        "=" * 65,
        "Noise Probe Report — Generator Listening Analysis",
        "=" * 65,
        "",
        f"Noisy trajektorie  : {summary['n_noisy_trajectories']}",
        f"Clean trajektorie  : {summary['n_clean_trajectories']}",
        f"Wspólne prompty    : {summary['n_shared_prompts']}",
        f"Embed model        : {summary['embed_model']}",
        "",
    ]

    def _section(title: str, d: Dict) -> None:
        lines.append(f"{'─' * 65}")
        lines.append(title)
        lines.append(f"{'─' * 65}")
        if "error" in d:
            lines.append(f"  ERROR: {d['error']}")
            lines.append("")
            return
        for k, v in d.items():
            if isinstance(v, float):
                lines.append(f"  {k:44s}: {v:.4f}")
            else:
                lines.append(f"  {k:44s}: {v}")
        # Werdykt
        sig = d.get("significant_at_005") or d.get("significant_at_005")
        if sig is not None:
            verdict = "ISTOTNA STATYSTYCZNIE (p<0.05)" if sig else "NIEISTOTNA (p≥0.05)"
            lines.append(f"  → Wynik: {verdict}")
        lines.append("")

    _section(
        "H1 — Nagroda końcowa  (Mann-Whitney U; clean > noisy)",
        summary["H1_reward"],
    )
    _section(
        "H2 — Cos-sim do gold-label  (Mann-Whitney U; clean > noisy)",
        summary["H2_cosine_similarity_to_gold"],
    )
    _section(
        "H3 — Dywergencja per prompt  (Wilcoxon signed-rank; div > 0)",
        summary["H3_hypothesis_divergence_per_prompt"],
    )

    lines += [
        "=" * 65,
        "Interpretacja wyników:",
        "",
        "  H1 istotna → generator z czystym kanałem osiąga wyższe nagrody.",
        "  H2 istotna → hipotezy clean są semantycznie bliżej ground truth.",
        "  H3 istotna → hipotezy clean i noisy różnią się dla tego samego",
        "               pytania → generator REAGUJE na treść kanału.",
        "",
        "  Brak istotności → generator może ignorować kanał retrievera,",
        "  albo szum nie degraduje komunikatu wystarczająco (sprawdź",
        "  noise_applied=True w trajektoriach i n_channel_tokens).",
        "=" * 65,
    ]

    report_path = os.path.join(output_dir, "noise_probe_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Raport tekstowy: {report_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Noise Probe — analiza statystyczna wpływu szumu na hipotezy generatora"
    )
    parser.add_argument(
        "--noisy_dir",
        required=True,
        help="Katalog z trajektoriami zaszumionymi (noise_probability=1.0). "
             "Może wskazywać na eval_noisy/ lub eval_noisy/eval/ — skrypt wykryje sam.",
    )
    parser.add_argument(
        "--clean_dir",
        required=True,
        help="Katalog z trajektoriami czystymi (noise_probability=0.0). "
             "Może wskazywać na eval_clean/ lub eval_clean/eval/.",
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_results/noise_probe",
        help="Gdzie zapisać wyniki analizy (default: ./eval_results/noise_probe)",
    )
    parser.add_argument(
        "--embed_host",
        default="localhost",
        help="Host serwera vLLM embedding (default: localhost)",
    )
    parser.add_argument(
        "--embed_port",
        type=int,
        default=8000,
        help="Port serwera vLLM embedding (default: 8000)",
    )
    parser.add_argument(
        "--embed_model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Nazwa modelu embedding na serwerze vLLM",
    )
    args = parser.parse_args()

    run_noise_probe_analysis(
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
        embed_host=args.embed_host,
        embed_port=args.embed_port,
        embed_model=args.embed_model,
    )