"""
extract_attention.py
====================
Skrypt do ekstrakcji i wizualizacji map uwagi (attention maps)
z modelu z pominięciem vLLM i FlashAttention.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# 1. Konfiguracja i dane wejściowe (TUTAJ WKLEJ SWOJE DANE)
# =============================================================================

MODEL_PATH = "Agents/models_output/grpo_qwen_5231830_results"
OUTPUT_DIR = "Agents/eval_results/attention_probe"

# Skopiuj z wybranego pliku traj_*.json czysty tekst komunikatu, 
# który wygenerował Retriever (najlepiej bez tagów <think>).
RETRIEVER_MESSAGE = """
The interpretation of decision processes in recurrent attention-based natural language inference (NLI) models requires advanced methods to probe intermediate alignment signals and recurrent states, especially when ordinary attention visualizations fail to provide meaningful insights. Papers highlight that attention mechanisms, such as those in DR-BiLSTM and ELMo, enable models to focus on critical semantic pairs and contextual features, but their interpretability is limited by standard visualization techniques. To address this, methods like Layer-wise Relevance Propagation (LRP) for RNNs offer a deterministic approach to attribute relevance to individual words, capturing multiplicative interactions and negation effects in a context-sensitive manner. Additionally, multi-perspective matching in BiMPM and dependent reading strategies in DR-BiLSTM enhance semantic alignment by modeling bidirectional and granular interactions between premise and hypothesis, respectively. These approaches collectively demonstrate that combining attention mechanisms with advanced attribution techniques and multi-granularity modeling can yield deeper insights into a model’s decision-making process. However, challenges remain in systematically analyzing recurrent states and alignment signals when standard methods fall short, necessitating further research into scalable and interpretable frameworks for deep semantic reasoning.
"""

# Skopiuj jedną, wybraną hipotezę, w której Generator odniósł się do faktów.
HYPOTHESIS = """
Integrating Layer-wise Relevance Propagation (LRP) for recurrent neural networks with multi-perspective matching and dependent reading strategies in recurrent attention-based natural language inference (NLI) models will enable systematic analysis of intermediate alignment signals and recurrent states, revealing the model’s decision-making process through context-sensitive attribution of relevance to semantic pairs and granular interactions between premise and hypothesis when ordinary attention visualizations fail to provide meaningful insights.
"""

# Zrekonstruowany pełny prompt (symulacja tego, co widział Generator).
# Zachowujemy strukturę ChatML.
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

# =============================================================================
# 2. Funkcje pomocnicze
# =============================================================================

def find_subsequence(full_list, sub_list):
    """Znajduje indeksy start i end podciągu w liście tokenów, z tolerancją na krawędziach."""
    # Odcinamy pierwszy i ostatni token (narażone na doklejenie do spacji/nowej linii)
    if len(sub_list) < 3:
        # Fallback dla bardzo krótkich tekstów
        n = len(sub_list)
        for i in range(len(full_list) - n + 1):
            if full_list[i:i+n] == sub_list:
                return i, i + n
        return None, None

    core = sub_list[1:-1]
    n = len(core)
    
    # Szukamy idealnego dopasowania "rdzenia"
    for i in range(len(full_list) - n + 1):
        if full_list[i:i+n] == core:
            # Zwracamy pozycję z uwzględnieniem odciętych krawędzi (-1 i +1)
            return i - 1, i + n + 1
            
    return None, None

# =============================================================================
# 3. Ładowanie Modelu (Z wymuszeniem Eager Attention)
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Ladowanie modelu z {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager" # KRYTYCZNE: Wyłącza FlashAttention!
    )
    
    print("Tokenizacja...")
    full_tokens = tokenizer.encode(FULL_TEXT, add_special_tokens=False)
    
    # Tokenizujemy fragmenty bez znaków specjalnych na brzegach, żeby łatwiej je znaleźć
    retriever_tokens = tokenizer.encode(RETRIEVER_MESSAGE.strip(), add_special_tokens=False)
    hypothesis_tokens = tokenizer.encode(HYPOTHESIS.strip(), add_special_tokens=False)
    
    # Szukamy granic naszych interesujących sekcji
    ret_start, ret_end = find_subsequence(full_tokens, retriever_tokens)
    hyp_start, hyp_end = find_subsequence(full_tokens, hypothesis_tokens)
    
    if ret_start is None or hyp_start is None:
        raise ValueError("Nie udało się zlokalizować podciągów w głównym tekście! Sprawdź formatowanie.")
        
    print(f"Retriever Message zlokalizowany: tokeny [{ret_start} - {ret_end}]")
    print(f"Hypothesis zlokalizowana: tokeny [{hyp_start} - {hyp_end}]")

    inputs = tokenizer(FULL_TEXT, return_tensors="pt").to(model.device)
    
    # =============================================================================
    # 4. Inferencja z wyciąganiem Wag Uwagi
    # =============================================================================
    print("Przeprowadzanie forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # outputs.attentions to tuple o rozmiarze równym liczbie warstw.
    # Wybieramy OSTATNIĄ warstwę, bo tam kształtuje się ostateczna decyzja semantyczna.
    # Kształt pojedynczej warstwy: (batch_size, num_heads, seq_len, seq_len)
    last_layer_attn = outputs.attentions[-1][0] 
    
    # Uśredniamy wagi po wszystkich głowach (heads) w ostatniej warstwie
    # Otrzymujemy matrycę (seq_len, seq_len)
    mean_attn = torch.mean(last_layer_attn, dim=0).float().cpu().numpy()
    
    # Wycinamy tylko fragment: Jak tokeny Hipotezy (oś Y) patrzyły na tokeny Retrievera (oś X)
    focus_matrix = mean_attn[hyp_start:hyp_end, ret_start:ret_end]
    
    # Przygotowujemy etykiety do osi
    y_labels = [tokenizer.decode([t]) for t in full_tokens[hyp_start:hyp_end]]
    x_labels = [tokenizer.decode([t]) for t in full_tokens[ret_start:ret_end]]

    # =============================================================================
    # 5. Rysowanie Heatmapy
    # =============================================================================
    print("Generowanie Heatmapy...")
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
    
    # Obrót etykiet dla czytelności
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    out_path = os.path.join(OUTPUT_DIR, "generator_attention_map.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Zakończono! Zapisano mapę pod: {out_path}")

if __name__ == "__main__":
    main()