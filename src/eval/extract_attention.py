"""
extract_attention.py
====================
Skrypt do ekstrakcji map uwagi. Generuje zarówno klasyczną 
heatmapę 2D (.png) jak i interaktywny dokument HTML z podświetlonym tekstem.
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

MODEL_PATH = "Agents/models_output/grpo_qwen_5232574_results"
OUTPUT_DIR = "Agents/eval_results/attention_probe"

RETRIEVER_MESSAGE = """The synthesis reveals that leveraging latent feature maps (AdvGAN++) and composite loss functions (AdvGAN, GAP) enhances adversarial generation efficiency and realism. Feature denoising blocks (Paper 2) suppress feature-level noise, preserving perceptual quality, while Rob-GAN's integration of adversarial training and GANs improves generalization. GAP's unified framework enables scalable, real-time attacks across tasks. These methods address efficiency, realism, and scalability, though challenges remain in balancing perturbation magnitude with perceptual fidelity and cross-model transferability."""

HYPOTHESIS = """1. Integrating latent feature maps with a composite loss function that balances adversarial training and perceptual fidelity, combined with feature denoising blocks to preserve realism, will enable efficient generation of targeted adversarial perturbations that scale across multiple classes and complex datasets."""

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
# 3. Inferencja i Generowanie HTML/PNG
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
        
    last_layer_attn = outputs.attentions[-1][0] 
    
    # Rzutowanie na float32 i konwersja do numpy (unika błędu bfloat16)
    mean_attn = torch.mean(last_layer_attn, dim=0).float().cpu().numpy()
    focus_matrix = mean_attn[hyp_start:hyp_end, ret_start:ret_end]
    
    # Etykiety do wykresu 2D
    y_labels = [tokenizer.decode([t]) for t in full_tokens[hyp_start:hyp_end]]
    x_labels = [tokenizer.decode([t]) for t in full_tokens[ret_start:ret_end]]

    # =============================================================================
    # 4. Budowa pliku PNG (Heatmapa 2D)
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
    
    plt.title("Uwaga Generatora skierowana na komunikat Retrievera (Ostatnia Warstwa, Średnia z Głów)", fontsize=16)
    plt.xlabel("Tokeny Komunikatu Retrievera", fontsize=14)
    plt.ylabel("Generowane Tokeny Hipotezy", fontsize=14)
    
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    out_path_png = os.path.join(OUTPUT_DIR, "generator_attention_map.png")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=300)
    print(f"Zapisano heatmapę pod: {out_path_png}")

    # =============================================================================
    # 5. Budowa pliku HTML (1D Cumulated Attention)
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
        
        f.write("<h2>Na co patrzył Generator?</h2>\n")
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