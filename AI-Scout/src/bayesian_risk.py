from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os

def main():
    print("==================================================")
    print("FASE 4: RETI BAYESIANE E ANALISI DEL RISCHIO")
    print("==================================================\n")
    
    # 1. Definizione della struttura (DAG)
    # FIX: Utilizzo di DiscreteBayesianNetwork al posto del deprecato BayesianNetwork
    model = DiscreteBayesianNetwork([
        ('Importanza_Match', 'Rischio_Malus'),
        ('Severita_Arbitro', 'Rischio_Malus')
    ])
    
    print("[1/3] Struttura del DAG creata con successo.")
    
    # 2. Definizione delle CPT (Conditional Probability Tables)
    
    # Nodo: Importanza_Match (0: Normale, 1: Derby/Scontro Diretto)
    cpd_importanza = TabularCPD(variable='Importanza_Match', variable_card=2, values=[[0.8], [0.2]])
    
    # Nodo: Severita_Arbitro (0: Bassa, 1: Alta)
    cpd_arbitro = TabularCPD(variable='Severita_Arbitro', variable_card=2, values=[[0.7], [0.3]])
    
    # Nodo: Rischio_Malus (0: Nessuno, 1: Giallo, 2: Rosso)
    cpd_malus = TabularCPD(
        variable='Rischio_Malus', variable_card=3,
        values=[
            [0.80, 0.50, 0.40, 0.15], # Probabilità Nessun Malus
            [0.18, 0.45, 0.50, 0.60], # Probabilità Cartellino Giallo
            [0.02, 0.05, 0.10, 0.25]  # Probabilità Cartellino Rosso
        ],
        evidence=['Importanza_Match', 'Severita_Arbitro'],
        evidence_card=[2, 2]
    )
    
    # Aggiunta delle CPT al modello e verifica
    model.add_cpds(cpd_importanza, cpd_arbitro, cpd_malus)
    assert model.check_model() == True
    print("[2/3] CPT caricate e validate matematicamente.\n")
    
    # 3. Inferenza Esatta (Variable Elimination)
    infer = VariableElimination(model)
    
    print("[3/3] Esecuzione Inferenza Esatta (Decision Support)...\n")
    
    # Scenario A: Partita normale con arbitro permissivo
    print(">>> SCENARIO A: Match Normale (0), Arbitro Permissivo (0)")
    risultato_A = infer.query(variables=['Rischio_Malus'], evidence={'Importanza_Match': 0, 'Severita_Arbitro': 0})
    print(risultato_A)
    
    # Scenario B: Derby infuocato con arbitro severissimo
    print("\n>>> SCENARIO B: Derby (1), Arbitro Severo (1)")
    risultato_B = infer.query(variables=['Rischio_Malus'], evidence={'Importanza_Match': 1, 'Severita_Arbitro': 1})
    print(risultato_B)
    
    print("\n==================================================")
    print(">>> SUPPORTO DECISIONALE:")
    print("Nel caso B, la probabilità di prendere un cartellino ROSSO")
    print("vola dal 2% al 25% (Rischio_Malus(2) = 0.2500).")
    print("Il Fanta-Manager dovrebbe valutare di lasciare il difensore in panchina!")
    print("==================================================")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()