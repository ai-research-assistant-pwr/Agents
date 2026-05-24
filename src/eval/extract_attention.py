"""
extract_attention.py
====================
Skrypt do ekstrakcji map uwagi. Generuje:
1. Wykres aktywności poszczególnych głów (per-head importance).
2. Klasyczną heatmapę 2D (.png) opartą na TOP K najaktywniejszych głowach.
3. Interaktywny dokument HTML z podświetlonym tekstem.
"""

import os
import torch
import numpy as np
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# 1. Konfiguracja i dane wejściowe
# =============================================================================

MODEL_PATH = "Agents/models_output/grpo_qwen_5231830_results"
OUTPUT_DIR = "Agents/eval_results/attention_probe"

RETRIEVER_MESSAGE = """The interpretation of decision processes in recurrent attention-based natural language inference (NLI) models requires advanced methods to probe intermediate alignment signals and recurrent states, especially when ordinary attention visualizations fail to provide meaningful insights. Papers highlight that attention mechanisms, such as those in DR-BiLSTM and ELMo, enable models to focus on critical semantic pairs and contextual features, but their interpretability is limited by standard visualization techniques. To address this, methods like Layer-wise Relevance Propagation (LRP) for RNNs offer a deterministic approach to attribute relevance to individual words, capturing multiplicative interactions and negation effects in a context-sensitive manner. Additionally, multi-perspective matching in BiMPM and dependent reading strategies in DR-BiLSTM enhance semantic alignment by modeling bidirectional and granular interactions between premise and hypothesis, respectively. These approaches collectively demonstrate that combining attention mechanisms with advanced attribution techniques and multi-granularity modeling can yield deeper insights into a model’s decision-making process. However, challenges remain in systematically analyzing recurrent states and alignment signals when standard methods fall short, necessitating further research into scalable and interpretable frameworks for deep semantic reasoning."""

HYPOTHESIS = """1. Integrating Layer-wise Relevance Propagation (LRP) for recurrent neural networks with multi-perspective matching and dependent reading strategies in recurrent attention-based natural language inference (NLI) models will enable systematic analysis of intermediate alignment signals and recurrent states, revealing the model’s decision-making process through context-sensitive attribution of relevance to semantic pairs and granular interactions between premise and hypothesis when ordinary attention visualizations fail to provide meaningful insights."""

FULL_TEXT = f"""<|im_start|>system
You are an Expert Research Scientist specializing in scientific hypothesis generation. You have received evidence syntheses from a retriever agent. Generate high-quality scientific hypotheses based on this evidence.

## Hypothesis Quality Criteria

Every hypothesis MUST satisfy ALL of the following:

1. **Clear and precisely stated** with well-defined variables and a proposed relationship or mechanism.
2. **Directly relevant** to the research query.
3. **Grounded in the provided evidence** — traceable to specific findings in the retriever's syntheses.
4. **Novel** — synthesize across findings, propose mechanistic explanations, or suggest extensions to new conditions.
5. **Testable and falsifiable** — a conceivable experiment could confirm or refute it.
6. **Multi-sentence** — describe the relationship, the proposed mechanism, and the conditions under which it holds. Split complex ideas across multiple sentences to improve clarity.
7. **Self-contained** — every newly introduced term, dataset, or method that is essential to understanding the hypothesis must be explained within the hypothesis itself. A domain expert reading only the hypothesis statement should be able to design an experiment to test it without needing additional context. After drafting each hypothesis, ask yourself: 'If I only had this statement, could I design an experiment to test it?' If the answer is no, revise to include the necessary clarifications.
8. **Specific, not generic** — avoid hypotheses that are too general or obvious. Focus on specific, non-trivial claims about mechanisms, interactions, or conditions.
9. **Method-focused when proposing new approaches** — if a hypothesis proposes a novel methodology or technique, it should describe the method itself in detail (what it does, how it integrates existing techniques, why it is expected to work), not just state that it achieves better performance on some task. For example, a good hypothesis would be: 'We propose method X that integrates techniques A and B in a unique way, allowing it to effectively leverage both structured and unstructured data for improved performance on task Y.'

Fewer well-grounded hypotheses are better than many speculative ones.

## Handling Channel Noise and [MASK] Tokens
The synthesis from the retriever agent may contain channel noise, specifically corrupted or missing words represented by the `[MASK]` token.
- **CRITICAL:** Do NOT attempt to guess, reconstruct, or fill in these missing words.
- Do NOT mention the `[MASK]` tokens, text corruption, or missing text in your thoughts or output.
- Treat `[MASK]` purely as irrelevant background noise. Focus your analytical attention entirely on the intact, readable scientific concepts and data surrounding the masks.

## Output Format

Output ONLY a numbered list of hypotheses, one per line, like:
1. [First hypothesis text]
2. [Second hypothesis text]
3. [Third hypothesis text]

Write only the hypothesis text — no preamble, no commentary, no leading phrases such as 'The hypothesis is:' or 'This paper proposes:'. Each hypothesis must stand on its own as a clean, concise declarative statement.
<|im_end|>
<|im_start|>user
Research query:
How can we better interpret the decision process of recurrent attention-based NLI models by probing their intermediate alignment signals and recurrent states, especially when ordinary attention visualizations are not informative?

[Initial retriever synthesis]
{RETRIEVER_MESSAGE}
<|im_end|>
<|im_start|>assistant
{HYPOTHESIS}"""

# =============================================================================
# 2. Funkcje pomocnicze
# =============================================================================

def find_subsequence(full_list, sub_list):
    """Znajduje indeksy start i end podciągu, uodpornione na tokeny brzegowe."""
    if len(sub_list) < 3:
        n = len(sub_list)
        for i in range(len(full_list) - n + 1):
            if full_list[i:i+n] == sub_list:
                return i, i + n
        return None, None

    core = sub_list[1:-1]
    n = len(core)
    for i in range(len(full_list) - n + 1):
        if full_list[i:i+n] == core:
            return i - 1, i + n + 1
            
    return None, None

# =============================================================================
# 3. Inferencja i Generowanie Wykresów
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Ladowanie modelu z {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager" 
    )
    
    print("Tokenizacja...")
    full_tokens = tokenizer.encode(FULL_TEXT, add_special_tokens=False)
    retriever_tokens = tokenizer.encode(RETRIEVER_MESSAGE.strip(), add_special_tokens=False)
    hypothesis_tokens = tokenizer.encode(HYPOTHESIS.strip(), add_special_tokens=False)
    
    ret_start, ret_end = find_subsequence(full_tokens, retriever_tokens)
    hyp_start, hyp_end = find_subsequence(full_tokens, hypothesis_tokens)
    
    if ret_start is None or hyp_start is None:
        raise ValueError("Nie udało się zlokalizować podciągów!")
        
    inputs = tokenizer(FULL_TEXT, return_tensors="pt").to(model.device)
    
    print("Przeprowadzanie forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Pobieramy 3 ostatnie warstwy i uśredniamy je, by złapać bogatszą semantykę (Odpowiedź na Komentarz 3)
    # Kształt: tuple -> stackujemy do (3, batch, heads, seq, seq) -> mean do (batch, heads, seq, seq)
    last_layers = torch.stack(outputs.attentions[-3:])
    pooled_attn = torch.mean(last_layers, dim=0)[0] # bierzemy batch=0 -> kształt (heads, seq, seq)
    
    # Wyciągamy uwagę z Hipotezy na Retrievera per głowa
    head_importance_matrix = pooled_attn[:, hyp_start:hyp_end, ret_start:ret_end] # (n_heads, hyp_len, ret_len)
    
    # Sumujemy całą uwagę rzuconą przez daną głowę (Odpowiedź na Komentarz 2)
    per_head_sum = head_importance_matrix.float().sum(dim=(1,2)).cpu().numpy()
    
    # =============================================================================
    # 4. Wykres: Aktywność Głów (Head Specialization)
    # =============================================================================
    print("Generowanie wykresu aktywności głów...")
    plt.figure(figsize=(10, 5))
    heads_x = np.arange(len(per_head_sum))
    plt.bar(heads_x, per_head_sum, color='steelblue')
    plt.title("Wyspecjalizowane głowy uwagi: Aktywność w relacji Hipoteza -> Komunikat")
    plt.xlabel("Indeks Głowy (Head ID)")
    plt.ylabel("Suma Wag Uwagi")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path_heads = os.path.join(OUTPUT_DIR, "head_activity_bar.png")
    plt.tight_layout()
    plt.savefig(out_path_heads, dpi=300)
    print(f"Zapisano wykres głów pod: {out_path_heads}")

    # =============================================================================
    # Wybieramy tylko TOP 3 najbardziej aktywne głowy do analizy (Zmniejszenie szumu)
    # =============================================================================
    top_k = 3
    top_heads_idx = np.argsort(per_head_sum)[-top_k:]
    print(f"Najbardziej aktywne 'retrieval heads' to: {top_heads_idx.tolist()}")
    
    # Filtrujemy macierz uwagi tylko do najlepszych głów i uśredniamy je
    best_heads_attn = head_importance_matrix[top_heads_idx]
    focus_matrix = torch.mean(best_heads_attn, dim=0).float().cpu().numpy()
    
    y_labels = [tokenizer.decode([t]) for t in full_tokens[hyp_start:hyp_end]]
    x_labels = [tokenizer.decode([t]) for t in full_tokens[ret_start:ret_end]]

    # =============================================================================
    # 5. Budowa pliku PNG (Heatmapa 2D z najlepszych głów)
    # =============================================================================
    print("Generowanie Heatmapy 2D...")
    plt.figure(figsize=(24, 16))
    
    sns.heatmap(
        focus_matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="Reds",
        cbar_kws={'label': 'Siła uwagi (Attention Weight)'}
    )
    
    plt.title(f"Uwaga Generatora na Retrievera (Uśrednione z TOP {top_k} najaktywniejszych głów, Ostatnie 3 warstwy)", fontsize=16)
    plt.xlabel("Tokeny Komunikatu Retrievera", fontsize=14)
    plt.ylabel("Generowane Tokeny Hipotezy", fontsize=14)
    
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    out_path_png = os.path.join(OUTPUT_DIR, "generator_attention_map.png")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300)
    print(f"Zapisano heatmapę pod: {out_path_png}")

    # =============================================================================
    # 6. Budowa pliku HTML (1D Cumulated Attention)
    # =============================================================================
    print("Obliczanie sumy uwagi (1D) dla HTML...")
    token_importance = np.sum(focus_matrix, axis=0)
    
    min_val = np.min(token_importance)
    max_val = np.max(token_importance)
    norm_importance = (token_importance - min_val) / (max_val - min_val + 1e-9)
    norm_importance = norm_importance ** 2 

    print("Generowanie dokumentu HTML...")
    html_path = os.path.join(OUTPUT_DIR, "retriever_attention.html")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><style>\n")
        f.write("body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.8; font-size: 18px; padding: 40px; max-width: 900px; margin: auto; background: #fafafa; color: #333; }\n")
        f.write(".token { border-radius: 3px; padding: 2px 0px; margin: 0; display: inline-block; }\n")
        f.write(".tooltip { position: relative; cursor: pointer; }\n")
        f.write(".tooltip .tooltiptext { visibility: hidden; width: max-content; background-color: rgba(0,0,0,0.8); color: #fff; text-align: center; border-radius: 4px; padding: 4px 8px; position: absolute; z-index: 1; bottom: 125%; left: 50%; transform: translateX(-50%); font-size: 13px; opacity: 0; transition: opacity 0.2s; }\n")
        f.write(".tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }\n")
        f.write("</style></head><body>\n")
        
        f.write(f"<h2>Na co patrzył Generator? (Wizualizacja z TOP {top_k} głów uwagi)</h2>\n")
        f.write("<p>Poniżej znajduje się komunikat Retrievera. Kolor czerwony oznacza fragmenty, z których Generator czerpał najwięcej informacji tworząc hipotezę nr 1. Najedź kursorem na wyraz, aby zobaczyć dokładną, skumulowaną wagę uwagi.</p>\n")
        f.write("<div style='background: white; padding: 30px; border: 1px solid #ddd; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: justify;'>\n")

        for i, token_id in enumerate(full_tokens[ret_start:ret_end]):
            word = tokenizer.decode([token_id])
            escaped_word = html.escape(word)
            score = norm_importance[i]
            
            r = 255
            g = int(255 * (1 - score))
            b = int(255 * (1 - score))
            
            f.write(f"<span class='token tooltip' style='background-color: rgb({r},{g},{b});'>")
            f.write(f"<pre style='display:inline; font-family:inherit; margin:0;'>{escaped_word}</pre>")
            f.write(f"<span class='tooltiptext'>Skumulowana waga: {token_importance[i]:.4f}</span>")
            f.write("</span>")

        f.write("\n</div></body></html>")
        
    print(f"Zapisano elegancki plik HTML pod: {html_path}")
    print("Zakończono pomyślnie!")

if __name__ == "__main__":
    main()