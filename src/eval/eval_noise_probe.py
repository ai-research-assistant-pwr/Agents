"""
eval_noise_probe.py — Generator Listening Probe: Analiza Statystyczna
======================================================================

Skrypt porównuje hipotezy wygenerowane w dwóch warunkach szumowych
(np. noise=0.0 vs noise=0.5, lub noise=0.0 vs noise=1.0) i odpowiada
na pytanie: czy generator faktycznie reaguje na treść komunikatu retrievera?

Wejście
-------
Dwa foldery z trajektoriami traj_*.json w formacie zapisywanym przez
scientific_workflow.py. Skrypt automatycznie wchodzi do podkatalogu eval/
jeśli taki istnieje (workflow zapisuje eval osobno gdy is_eval=True).

Struktura oczekiwanego JSON:
  {
    "prompt_id": 23,
    "prompt": "...",
    "label": "...",
    "total_reward": 2.34,
    "ec_stats": { "apply_channel_noise": true, ... },
    "turns": [
      ...,
      {
        "turn_type": "generator_generate",
        "hypotheses": ["hyp1", "hyp2", ...],
        "similarity_score": 0.84,
        ...
      }
    ]
  }

Kontrola wstępna
----------------
Przed analizą skrypt drukuje tabelę kontrolną: ile trajektorii per prompt_id
znaleziono w każdym folderze. Pozwala to wykryć brakujące lub zdublowane próbki
zanim przejdziemy do testów.

Testowane hipotezy
------------------
  H1 — NAGRODA (total_reward):
       Mann-Whitney U jednostronny: folder_a > folder_b
       Obserwacja: każda trajektoria osobno

  H2 — COS-SIM DO GOLD LABEL:
       cos(embedding_hipotezy, embedding_labela) per trajektoria
       Mann-Whitney U jednostronny: folder_a > folder_b

  H3 — DYWERGENCJA HIPOTEZ WEWNĄTRZ PROMPT_ID:
       Dla każdego prompt_id: średnia odległość cosinus między
       wszystkimi parami (hyp_a_i, hyp_b_j).
       Wilcoxon signed-rank jednostronny vs 0 (div > 0)
       Obserwacja: jeden punkt per prompt_id (odporność na pseudoreplikację)

Output
------
  <output_dir>/
    ├── noise_probe_report.txt
    ├── noise_probe_summary.json
    ├── noise_probe_rewards.png
    ├── noise_probe_similarity.png
    └── noise_probe_divergence.png

Użycie
------
  python src/eval/eval_noise_probe.py \\
      --dir_a   Agents/workflow_logs/noise_probe/eval_clean \\
      --dir_b   Agents/workflow_logs/noise_probe/eval_noisy \\
      --label_a "clean (noise=0.0)" \\
      --label_b "noisy (noise=0.5)" \\
      --output_dir Agents/eval_results/noise_probe \\
      [--embed_host localhost] [--embed_port 8000] \\
      [--embed_model Qwen/Qwen3-Embedding-4B]
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


# =============================================================================
# Stałe stylu
# =============================================================================

COLOR_A   = "#4C72B0"
COLOR_B   = "#DD8452"
COLOR_REF = "#aaaaaa"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# Ładowanie trajektorii
# =============================================================================

def _resolve_dir(directory: str) -> str:
    """
    Jeśli w katalogu nie ma traj_*.json, ale jest podkatalog eval/ —
    wchodzi do niego (workflow zapisuje eval do <debug_dir>/eval/).
    """
    if glob.glob(os.path.join(directory, "traj_*.json")):
        return directory
    eval_sub = os.path.join(directory, "eval")
    if os.path.isdir(eval_sub) and glob.glob(os.path.join(eval_sub, "traj_*.json")):
        print(f"  [INFO] Auto-wykryto podkatalog /eval/ → {eval_sub}")
        return eval_sub
    return directory  # błąd wyjdzie przy load


def _get_hypothesis_text(turns: List[Dict]) -> str:
    """Wyciąga tekst hipotez z ostatniego kroku generator_generate."""
    for t in reversed(turns):
        if t.get("turn_type") == "generator_generate":
            hyps = t.get("hypotheses", [])
            if isinstance(hyps, list) and hyps:
                return " ".join(str(h).strip() for h in hyps)
            return t.get("output", "").strip()
    return ""


def load_trajectories(directory: str) -> List[Dict]:
    """
    Wczytuje wszystkie traj_*.json z katalogu.
    Zwraca listę rekordów z kluczami:
      prompt_id, prompt, label, total_reward, hypothesis_text, noise_applied
    """
    resolved = _resolve_dir(directory)
    files = sorted(glob.glob(os.path.join(resolved, "traj_*.json")))
    if not files:
        files = sorted(glob.glob(os.path.join(resolved, "**", "traj_*.json"), recursive=True))

    records = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            ec = data.get("ec_stats", {})
            records.append({
                "file":            fp,
                "prompt_id":       data.get("prompt_id", -1),
                "prompt":          data.get("prompt", ""),
                "label":           data.get("label", ""),
                "total_reward":    float(data.get("total_reward", 0.0)),
                "hypothesis_text": _get_hypothesis_text(data.get("turns", [])),
                "noise_applied":   bool(ec.get("apply_channel_noise", False)),
            })
        except Exception as exc:
            print(f"  [WARN] Nie można wczytać {fp}: {exc}")

    return records


# =============================================================================
# Kontrola wstępna: liczba trajektorii per prompt_id
# =============================================================================

def check_trajectory_counts(
    recs_a: List[Dict],
    recs_b: List[Dict],
    label_a: str,
    label_b: str,
) -> Tuple[bool, Dict[int, int], Dict[int, int]]:
    """
    Drukuje tabelę: prompt_id | count_a | count_b | status.
    Zwraca (ok, counts_a, counts_b).
    ok = True jeśli każdy prompt_id ma taką samą, niezerową liczbę próbek w obu folderach.
    """
    counts_a: Dict[int, int] = defaultdict(int)
    counts_b: Dict[int, int] = defaultdict(int)
    for r in recs_a:
        counts_a[r["prompt_id"]] += 1
    for r in recs_b:
        counts_b[r["prompt_id"]] += 1

    all_ids = sorted(set(counts_a) | set(counts_b))

    # Skrócona tabela: tylko ID z problemem + podsumowanie
    mismatches = []
    for pid in all_ids:
        ca = counts_a.get(pid, 0)
        cb = counts_b.get(pid, 0)
        if ca != cb or ca == 0:
            mismatches.append(pid)

    ok = len(mismatches) == 0

    # Zawsze drukuj liczby zbiorcze
    print(f"\n  {'─'*62}")
    print(f"  Kontrola: liczba trajektorii per prompt_id")
    print(f"  {'─'*62}")

    # Wyznacz typową liczbę próbek (moda counts_a dla wspólnych ID)
    shared_ids = sorted(set(counts_a) & set(counts_b))
    if shared_ids:
        typical = max(set(counts_a[p] for p in shared_ids),
                      key=list(counts_a[p] for p in shared_ids).count)
    else:
        typical = 0

    print(f"  Prompt_id łącznie   : {len(all_ids)}")
    print(f"  Prompt_id wspólnych : {len(shared_ids)}")
    print(f"  Typowa l. próbek    : {typical} (per folder)")
    print(f"  Trajektorie A       : {len(recs_a)}  ({label_a})")
    print(f"  Trajektorie B       : {len(recs_b)}  ({label_b})")

    if mismatches:
        print(f"\n  [WARN] Niezgodna liczba próbek dla {len(mismatches)} prompt_id:")
        print(f"  {'prompt_id':>10}  {label_a:>20}  {label_b:>20}  {'status':>8}")
        print(f"  {'─'*10}  {'─'*20}  {'─'*20}  {'─'*8}")
        for pid in mismatches:
            ca = counts_a.get(pid, 0)
            cb = counts_b.get(pid, 0)
            print(f"  {pid:>10}  {ca:>20}  {cb:>20}  {'BŁĄD':>8}")
        print(f"\n  Analiza będzie oparta tylko na {len(shared_ids)} wspólnych prompt_id.")
    else:
        print(f"  [OK] Wszystkie prompt_id mają {typical} próbek w obu folderach.")

    print(f"  {'─'*62}")

    return ok, dict(counts_a), dict(counts_b)


# =============================================================================
# Embeddingi
# =============================================================================

def _embed_vllm(texts: List[str], host: str, port: int, model: str) -> Optional[np.ndarray]:
    try:
        import requests
        resp = requests.post(
            f"http://{host}:{port}/v1/embeddings",
            json={"model": model, "input": texts},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array([item["embedding"] for item in data["data"]], dtype=np.float32)
    except Exception as exc:
        print(f"  [WARN] vLLM embedding error: {exc}")
        return None


def _embed_st(texts: List[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return m.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    except ImportError:
        raise RuntimeError(
            "Serwer vLLM niedostępny i brak sentence-transformers.\n"
            "Zainstaluj: pip install sentence-transformers"
        )


def get_embeddings(
    texts: List[str],
    host: str,
    port: int,
    model: str,
    batch: int = 64,
) -> np.ndarray:
    parts = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        emb = _embed_vllm(chunk, host, port, model)
        if emb is None:
            print("  [INFO] Fallback: sentence-transformers (all-MiniLM-L6-v2)")
            return _embed_st(texts)
        parts.append(emb)
    return np.vstack(parts)


def cos_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cos-similarity row-by-row, obie macierze (n, d)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.einsum("ij,ij->i", an, bn)


def cos_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pełna macierz podobieństwa (n_a, n_b)."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return an @ bn.T


# =============================================================================
# Testy statystyczne
# =============================================================================

def mwu(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    metric: str,
    alternative: str = "greater",
) -> Dict:
    """Mann-Whitney U (domyślnie H₁: median(a) > median(b))."""
    stat, p = mannwhitneyu(a, b, alternative=alternative)
    # rank-biserial effect size ∈ [-1, 1]
    r = (2 * stat / (len(a) * len(b))) - 1
    return {
        "test":              "Mann-Whitney U",
        "metric":            metric,
        "alternative":       f"{label_a} > {label_b}",
        "n_a":               int(len(a)),
        "n_b":               int(len(b)),
        "mean_a":            float(np.mean(a)),
        "mean_b":            float(np.mean(b)),
        "median_a":          float(np.median(a)),
        "median_b":          float(np.median(b)),
        "U_statistic":       float(stat),
        "p_value":           float(p),
        "effect_r_biserial": float(r),
        "significant_005":   bool(p < 0.05),
        "significant_001":   bool(p < 0.01),
    }


def wilcoxon_gt0(values: np.ndarray, metric: str) -> Dict:
    """
    Wilcoxon signed-rank jednostronny: H₁: median > 0.
    Testuje czy dywergencja między warunkami jest istotnie > 0.
    """
    try:
        stat, p = wilcoxon(values, alternative="greater")
    except TypeError:
        # fallback dla starszego scipy bez parametru alternative
        stat, p2 = wilcoxon(values)
        p = p2 / 2 if float(np.mean(values)) > 0 else 1.0 - p2 / 2

    return {
        "test":              "Wilcoxon signed-rank",
        "metric":            metric,
        "alternative":       "median > 0",
        "n_prompt_ids":      int(len(values)),
        "mean":              float(np.mean(values)),
        "median":            float(np.median(values)),
        "std":               float(np.std(values)),
        "W_statistic":       float(stat),
        "p_value_one_sided": float(p),
        "significant_005":   bool(p < 0.05),
        "significant_001":   bool(p < 0.01),
    }


# =============================================================================
# Wykresy
# =============================================================================

def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def plot_boxplot(
    data_a: np.ndarray,
    data_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    path: str,
    p_value: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bp = ax.boxplot(
        [data_a, data_b],
        labels=[label_a, label_b],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.35, linestyle="none"),
    )
    for patch, color in zip(bp["boxes"], [COLOR_A, COLOR_B]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", lw=0.4, ls=":", color=COLOR_REF)

    if p_value is not None:
        stars = _sig_stars(p_value)
        combined = np.concatenate([data_a, data_b])
        y_top = float(np.percentile(combined, 97)) if len(combined) else 1.0
        y_ann = y_top + abs(y_top) * 0.07
        ax.annotate(stars, xy=(1.5, y_ann), ha="center", fontsize=14, color="black")
        ax.annotate(
            f"p={p_value:.3f}", xy=(1.5, y_ann),
            xytext=(0, -14), textcoords="offset points",
            ha="center", fontsize=8, color="#444",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Wykres: {path}")


def plot_divergence(
    values: np.ndarray,
    title: str,
    path: str,
    p_value: Optional[float] = None,
) -> None:
    """Boxplot dywergencji per prompt_id z linią referencyjną zero."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    bp = ax.boxplot(
        values,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markersize=3, alpha=0.35, linestyle="none"),
    )
    bp["boxes"][0].set_facecolor("#8172B2")
    bp["boxes"][0].set_alpha(0.75)

    ax.axhline(0, color="red", lw=1.2, ls="--", label="zero (brak dywergencji)")
    ax.set_xticklabels(["Per-prompt_id\ndywergencja"])
    ax.set_ylabel("Średnia cosine distance\n(1 − cos(hyp_a, hyp_b))")
    ax.set_title(title, pad=8)
    ax.grid(axis="y", lw=0.4, ls=":", color=COLOR_REF)
    ax.legend(frameon=False, fontsize=8)

    if p_value is not None:
        stars = _sig_stars(p_value)
        med = float(np.median(values))
        ax.annotate(
            f"{stars}  p={p_value:.3f}",
            xy=(1, med), xytext=(12, 6), textcoords="offset points",
            fontsize=9, color="black",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Wykres: {path}")


# =============================================================================
# Główna funkcja
# =============================================================================

def run(
    dir_a: str,
    dir_b: str,
    label_a: str,
    label_b: str,
    output_dir: str,
    embed_host: str,
    embed_port: int,
    embed_model: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Wczytaj trajektorie ────────────────────────────────────────────────
    print(f"\n[1/5] Wczytywanie trajektorii...")
    print(f"  A ({label_a}): {dir_a}")
    print(f"  B ({label_b}): {dir_b}")

    recs_a = load_trajectories(dir_a)
    recs_b = load_trajectories(dir_b)

    if not recs_a:
        sys.exit(f"ERROR: Brak trajektorii w {dir_a}")
    if not recs_b:
        sys.exit(f"ERROR: Brak trajektorii w {dir_b}")

    print(f"  Wczytano: {len(recs_a)} ({label_a}),  {len(recs_b)} ({label_b})")

    # ── 2. Kontrola próbek per prompt_id ─────────────────────────────────────
    print(f"\n[2/5] Kontrola próbek per prompt_id...")
    counts_ok, counts_a, counts_b = check_trajectory_counts(recs_a, recs_b, label_a, label_b)

    # Analiza tylko na wspólnych prompt_id
    shared_ids = set(counts_a) & set(counts_b)
    recs_a = [r for r in recs_a if r["prompt_id"] in shared_ids]
    recs_b = [r for r in recs_b if r["prompt_id"] in shared_ids]

    if not recs_a or not recs_b:
        sys.exit("ERROR: Żadnych wspólnych prompt_id między folderami.")

    # ── 3. H1: Nagroda ───────────────────────────────────────────────────────
    print(f"\n[3/5] H1 — Nagroda (Mann-Whitney U)...")
    rew_a = np.array([r["total_reward"] for r in recs_a])
    rew_b = np.array([r["total_reward"] for r in recs_b])

    h1 = mwu(rew_a, rew_b, label_a, label_b, "total_reward", alternative="greater")
    _print_result(f"H1 — total_reward  ({label_a} > {label_b})", h1)

    # ── 4. Embeddingi + H2 ───────────────────────────────────────────────────
    print(f"\n[4/5] Obliczanie embeddingów i H2 — cos-sim do gold-label...")

    valid_a = [r for r in recs_a if r["hypothesis_text"].strip()]
    valid_b = [r for r in recs_b if r["hypothesis_text"].strip()]

    if not valid_a or not valid_b:
        sys.exit("ERROR: Za mało niepustych hipotez do analizy semantycznej.")

    n_va, n_vb = len(valid_a), len(valid_b)

    # Jedna paczka → jedno wywołanie serwera
    all_texts = (
        [r["hypothesis_text"] for r in valid_a]   # 0 .. n_va-1
        + [r["hypothesis_text"] for r in valid_b] # n_va .. n_va+n_vb-1
        + [r["label"] for r in valid_a]           # n_va+n_vb .. 2*n_va+n_vb-1
        + [r["label"] for r in valid_b]           # ostatnie n_vb
    )

    print(f"  Embedduję {len(all_texts)} tekstów ({n_va}+{n_vb} hipotez, {n_va}+{n_vb} labeli)...")
    embs = get_embeddings(all_texts, embed_host, embed_port, embed_model)

    emb_hyp_a = embs[:n_va]
    emb_hyp_b = embs[n_va : n_va + n_vb]
    emb_lab_a = embs[n_va + n_vb : n_va + n_vb + n_va]
    emb_lab_b = embs[n_va + n_vb + n_va :]

    sim_a = cos_rowwise(emb_hyp_a, emb_lab_a)
    sim_b = cos_rowwise(emb_hyp_b, emb_lab_b)

    h2 = mwu(sim_a, sim_b, label_a, label_b, "cosine_similarity_to_gold", alternative="greater")
    _print_result(f"H2 — cos-sim do gold-label  ({label_a} > {label_b})", h2)

    # ── 5. H3: Dywergencja per prompt_id ──────────────────────────────────────
    print(f"\n[5/5] H3 — Dywergencja hipotez per prompt_id (Wilcoxon)...")

    # Indeks: prompt_id → lista indeksów w tablicach emb_hyp_a / emb_hyp_b
    idx_a: Dict[int, List[int]] = defaultdict(list)
    idx_b: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(valid_a):
        idx_a[r["prompt_id"]].append(i)
    for i, r in enumerate(valid_b):
        idx_b[r["prompt_id"]].append(i)

    shared_valid_ids = sorted(set(idx_a) & set(idx_b))
    per_prompt_div: List[float] = []

    for pid in shared_valid_ids:
        ea = emb_hyp_a[idx_a[pid]]   # (n_samples_a, d)
        eb = emb_hyp_b[idx_b[pid]]   # (n_samples_b, d)
        # Macierz wszystkich par → średnia odległość cosinus
        dist = 1.0 - cos_matrix(ea, eb)  # (n_a, n_b)
        per_prompt_div.append(float(np.mean(dist)))

    div_arr = np.array(per_prompt_div)
    print(f"  Prompt_id z parowanymi hipotezami: {len(div_arr)}")

    if len(div_arr) < 2:
        print("  [WARN] Za mało prompt_id — pomijam test Wilcoxona.")
        h3: Dict = {"error": "za mało prompt_id", "n_prompt_ids": int(len(div_arr))}
    else:
        h3 = wilcoxon_gt0(div_arr, "cosine_distance_per_prompt_id")
        _print_result("H3 — dywergencja hipotez per prompt_id", h3)

    # ── Zapis JSON ────────────────────────────────────────────────────────────
    summary = {
        "dir_a":                  dir_a,
        "dir_b":                  dir_b,
        "label_a":                label_a,
        "label_b":                label_b,
        "embed_model":            embed_model,
        "n_trajectories_a":       len(recs_a),
        "n_trajectories_b":       len(recs_b),
        "n_shared_prompt_ids":    len(shared_ids),
        "counts_per_prompt_id_a": counts_a,
        "counts_per_prompt_id_b": counts_b,
        "H1_reward":                       h1,
        "H2_cosine_similarity_to_gold":    h2,
        "H3_divergence_per_prompt_id":     h3,
    }

    json_path = os.path.join(output_dir, "noise_probe_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {json_path}")

    _write_report(summary, output_dir)

    # ── Wykresy ───────────────────────────────────────────────────────────────
    plot_boxplot(
        rew_a, rew_b, label_a, label_b,
        title=f"H1 — Nagroda: {label_a} vs {label_b}",
        ylabel="total_reward",
        path=os.path.join(output_dir, "noise_probe_rewards.png"),
        p_value=h1["p_value"],
    )

    plot_boxplot(
        sim_a, sim_b, label_a, label_b,
        title=f"H2 — Cos-sim do gold-label:\n{label_a} vs {label_b}",
        ylabel="Cosine similarity",
        path=os.path.join(output_dir, "noise_probe_similarity.png"),
        p_value=h2["p_value"],
    )

    if "error" not in h3 and len(div_arr) >= 2:
        plot_divergence(
            div_arr,
            title=f"H3 — Dywergencja hipotez per prompt_id\n({label_a} vs {label_b})",
            path=os.path.join(output_dir, "noise_probe_divergence.png"),
            p_value=h3.get("p_value_one_sided"),
        )

    print(f"\n[✓] Gotowe. Wyniki: {output_dir}")


# =============================================================================
# Pomocnicze drukowanie
# =============================================================================

def _print_result(heading: str, result: Dict) -> None:
    W = 62
    print(f"\n  {'─'*W}")
    print(f"  {heading}")
    print(f"  {'─'*W}")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k:44s}: {v:.4f}")
        else:
            print(f"  {k:44s}: {v}")
    sig = result.get("significant_005")
    if sig is not None:
        verdict = "ISTOTNA STATYSTYCZNIE (p<0.05)" if sig else "NIEISTOTNA (p≥0.05)"
        print(f"  → Wynik: {verdict}")


def _write_report(summary: Dict, output_dir: str) -> None:
    la, lb = summary["label_a"], summary["label_b"]
    lines = [
        "=" * 65,
        "Noise Probe Report",
        f"  A: {la}  ({summary['dir_a']})",
        f"  B: {lb}  ({summary['dir_b']})",
        "=" * 65,
        "",
        f"  Trajektorie A      : {summary['n_trajectories_a']}",
        f"  Trajektorie B      : {summary['n_trajectories_b']}",
        f"  Wspólne prompt_id  : {summary['n_shared_prompt_ids']}",
        f"  Embed model        : {summary['embed_model']}",
        "",
    ]

    def sec(title: str, d: Dict) -> None:
        lines.append(f"{'─'*65}")
        lines.append(title)
        lines.append(f"{'─'*65}")
        if "error" in d:
            lines.append(f"  ERROR: {d['error']}")
            lines.append("")
            return
        for k, v in d.items():
            if isinstance(v, float):
                lines.append(f"  {k:44s}: {v:.4f}")
            else:
                lines.append(f"  {k:44s}: {v}")
        sig = d.get("significant_005")
        if sig is not None:
            lines.append(f"  → {'ISTOTNA (p<0.05)' if sig else 'NIEISTOTNA (p≥0.05)'}")
        lines.append("")

    sec(f"H1 — total_reward  (MWU; {la} > {lb})", summary["H1_reward"])
    sec(f"H2 — cos-sim do gold-label  (MWU; {la} > {lb})",
        summary["H2_cosine_similarity_to_gold"])
    sec(f"H3 — dywergencja per prompt_id  (Wilcoxon; div > 0)",
        summary["H3_divergence_per_prompt_id"])

    lines += [
        "=" * 65,
        "Interpretacja:",
        "  H1+H2 istotne → generator z czystszym kanałem osiąga wyższe",
        "    nagrody i produkuje hipotezy bliższe ground-truth.",
        "  H3 istotna → hipotezy dla tego samego pytania różnią się",
        "    semantycznie w zależności od szumu → generator REAGUJE",
        "    na treść komunikatu retrievera.",
        "  Brak istotności → generator może ignorować kanał lub",
        "    szum nie degraduje komunikatu wystarczająco.",
        "=" * 65,
    ]

    path = os.path.join(output_dir, "noise_probe_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Raport: {path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Noise Probe — test statystyczny wpływu szumu kanału na hipotezy generatora"
    )
    parser.add_argument("--dir_a", required=True,
        help="Folder z trajektoriami warunku A (np. eval_clean)")
    parser.add_argument("--dir_b", required=True,
        help="Folder z trajektoriami warunku B (np. eval_noisy lub eval_half)")
    parser.add_argument("--label_a", default="A",
        help="Etykieta warunku A na wykresach (np. 'clean (noise=0.0)')")
    parser.add_argument("--label_b", default="B",
        help="Etykieta warunku B na wykresach (np. 'noisy (noise=0.5)')")
    parser.add_argument("--output_dir", default="./eval_results/noise_probe",
        help="Katalog wyjściowy (default: ./eval_results/noise_probe)")
    parser.add_argument("--embed_host", default="localhost")
    parser.add_argument("--embed_port", type=int, default=8000)
    parser.add_argument("--embed_model", default="Qwen/Qwen3-Embedding-4B")

    args = parser.parse_args()

    run(
        dir_a=args.dir_a,
        dir_b=args.dir_b,
        label_a=args.label_a,
        label_b=args.label_b,
        output_dir=args.output_dir,
        embed_host=args.embed_host,
        embed_port=args.embed_port,
        embed_model=args.embed_model,
    )