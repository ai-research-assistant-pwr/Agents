import os
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Konfiguracja stylu (identyczna z eksperymentami z morfologii)
# =============================================================================
_STYLE = {
    'font.family':      'serif',
    'font.size':        11,
    'axes.labelsize':   11,
    'axes.titlesize':   12,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'axes.linewidth':   0.8,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
}

def generate_plots_from_csv(csv_path: str, output_dir: str, suffix: str, title_desc: str):
    """Wczytuje plik CSV i generuje wykresy ugruntowania oraz dyspersji."""
    
    if not os.path.exists(csv_path):
        print(f"[OSTRZEŻENIE] Brak pliku: {csv_path}")
        return
        
    print(f"Przetwarzanie pliku: {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    plt.rcParams.update(_STYLE)
    GREY_REF = '#cccccc'
    
    # -------------------------------------------------------------------------
    # 1. Wykres: Dyspersja sygnału
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    
    for label, group in df.groupby("Run"):
        ax1.plot(group["Window_Index"], group["Mean_Dispersion"], label=label, lw=1.5)
        ax1.fill_between(group["Window_Index"], 
                         group["Mean_Dispersion"] - group["Std_Dispersion"], 
                         group["Mean_Dispersion"] + group["Std_Dispersion"], alpha=0.1)
    
    ax1.set_title(f"Dyspersja sygnału wewnątrz okna ({title_desc})", pad=10)
    ax1.set_xlabel("Okno treningowe")
    ax1.set_ylabel("Średni dystans kosinusowy")
    ax1.grid(axis='y', lw=0.4, color=GREY_REF, ls='--')
    
    # Legenda wyciągnięta na prawą stronę
    # ax1.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    out_disp = os.path.join(output_dir, f"exp2_dispersion_over_time_{suffix}.png")
    plt.tight_layout()
    plt.savefig(out_disp, dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Wykres: Ugruntowanie (Grounding)
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    
    for label, group in df.groupby("Run"):
        ax2.plot(group["Window_Index"], group["Mean_Grounding"], label=label, lw=1.5)
        ax2.fill_between(group["Window_Index"], 
                         group["Mean_Grounding"] - group["Std_Grounding"], 
                         group["Mean_Grounding"] + group["Std_Grounding"], alpha=0.1)
        
    ax2.set_title(f"Ugruntowanie - Podobieństwo do zapytania ({title_desc})", pad=10)
    ax2.set_xlabel("Okno treningowe")
    ax2.set_ylabel("Podobieństwo kosinusowe")
    ax2.grid(axis='y', lw=0.4, color=GREY_REF, ls='--')
    
    # Legenda wyciągnięta na prawą stronę
    # ax2.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    out_ground = os.path.join(output_dir, f"exp2_grounding_over_time_{suffix}.png")
    plt.tight_layout()
    plt.savefig(out_ground, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [✓] Zapisano wykresy dla: {title_desc} (_{suffix}.png)")


if __name__ == "__main__":
    # =========================================================================
    # DYNAMICZNE ŚCIEŻKI
    # =========================================================================
    # Wyliczamy ścieżkę do głównego folderu "Agents" poprzez cofnięcie się
    # o 3 poziomy w górę od lokalizacji tego skryptu:
    # eval_semantics.py -> grpo -> scripts -> Agents
    CURRENT_FILE = os.path.abspath(__file__)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
    
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "semantics_csvs")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "semantics_plots")
    
    # Upewniamy się, że foldery istnieją
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"\n[BŁĄD] Folder wejściowy nie istnieje: {INPUT_DIR}")
        print("Upewnij się, że struktura to: Agents/data/semantics_csvs/")
        exit(1)
        
    # Definicja 3 wariantów eksperymentu
    experiments = [
        ("exp2_signal_structure_1.csv", "1", "Bez kompresji"),
        ("exp2_signal_structure_2.csv", "2", "Kompresja umiarkowana"),
        ("exp2_signal_structure_3.csv", "3", "Kompresja ekstremalna")
    ]
    
    print("="*60)
    print("Wizualizacja eksperymentu z semantyki (z plików CSV)")
    print(f"Odczyt z: {INPUT_DIR}")
    print(f"Zapis do: {OUTPUT_DIR}")
    print("="*60)
    
    for filename, suffix, title_desc in experiments:
        csv_path = os.path.join(INPUT_DIR, filename)
        generate_plots_from_csv(csv_path, OUTPUT_DIR, suffix, title_desc)
        
    print("\n[SUKCES] Zakończono generowanie wykresów.")