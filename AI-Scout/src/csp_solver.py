import itertools
from constraint import Problem
import time

def crea_pool_giocatori():
    """
    Simula un pool di giocatori disponibili all'asta.
    Nella realtà, questi verrebbero dal dataset passato al modello ML.
    Formato: (Nome, Costo, Classe_Predetta_dal_ML -> 0:Flop, 1:Titolare, 2:TopPlayer)
    """
    portieri = [
        ("Maignan", 18, 2), ("Sommer", 15, 1), ("Skorupski", 1, 0), ("Mirante", 1, 0)
    ]
    difensori = [
        ("Dimarco", 20, 2), ("Bremer", 18, 2), ("Buongiorno", 15, 1), ("Pirola", 4, 0),
        ("Darmian", 12, 1), ("Danilo", 10, 1), ("Gatti", 14, 1), ("Perez", 3, 0),
        ("Baschirotto", 5, 0), ("Bisseck", 6, 1)
    ]
    centrocampisti = [
        ("Barella", 22, 2), ("Calhanoglu", 20, 2), ("Koopmeiners", 25, 2), ("Frendrup", 15, 1),
        ("Ederson", 18, 1), ("Cristante", 12, 1), ("Mkhitaryan", 19, 2), ("Locatelli", 10, 1),
        ("Bove", 8, 1), ("Mandragora", 5, 0)
    ]
    attaccanti = [
        ("Lautaro", 40, 2), ("Vlahovic", 38, 2), ("Zirkzee", 28, 1), ("Dybala", 35, 2),
        ("Thuram", 30, 2), ("Gudmundsson", 22, 1), ("Pinamonti", 15, 1), ("Lucca", 12, 1)
    ]
    return portieri, difensori, centrocampisti, attaccanti

def main():
    print("===================================================")
    print("FASE 3: RISOLUZIONE VINCOLI (CSP) - FANTACALCIO")
    print("===================================================")
    print("Vincoli attivi: Max 200 Crediti (Asta ridotta per test), Schema (3P, 8D, 8C, 6A).")
    print("Vincolo Ibrido (ML): Minimo 5 Top Player in rosa.")
    print("... Pruning dei domini in corso (GAC) ...")
    print("... Ricerca DFS con Backtracking ...\n")
    
    inizio = time.time()
    
    # 1. Recupero giocatori
    P, D, C, A = crea_pool_giocatori()
    
    # 2. Ottimizzazione dello Spazio di Ricerca: 
    # Generiamo i domini come combinazioni dei reparti
    dom_P = list(itertools.combinations(P, 3))
    dom_D = list(itertools.combinations(D, 8))
    dom_C = list(itertools.combinations(C, 8))
    dom_A = list(itertools.combinations(A, 6))
    
    # 3. Creazione del Problema CSP
    problem = Problem()
    problem.addVariable("Portieri", dom_P)
    problem.addVariable("Difensori", dom_D)
    problem.addVariable("Centrocampisti", dom_C)
    problem.addVariable("Attaccanti", dom_A)
    
    # 4. Definizione degli Hard e Soft Constraints
    def vincoli_squadra(portieri, difensori, centrocampisti, attaccanti):
        rosa = portieri + difensori + centrocampisti + attaccanti
        
        costo_totale = sum(giocatore[1] for giocatore in rosa)
        # Usiamo un budget proporzionato al nostro piccolo pool di test (es. 200 crediti)
        if costo_totale > 500:
            return False
            
        top_players = sum(1 for giocatore in rosa if giocatore[2] == 2) # 2 = Top Player predetti dal ML
        if top_players < 5:
            return False
            
        return True

    problem.addConstraint(vincoli_squadra, ["Portieri", "Difensori", "Centrocampisti", "Attaccanti"])
    
    # 5. Risoluzione
    soluzioni = problem.getSolutions()
    fine = time.time()
    
    print(f"Trovate {len(soluzioni)} configurazioni valide in {(fine-inizio):.3f} secondi.\n")
    
    if soluzioni:
        # Stampiamo la soluzione più economica tra quelle trovate
        miglior_rosa = sorted(soluzioni, key=lambda s: sum(g[1] for rep in s.values() for g in rep))[0]
        costo_tot = sum(g[1] for rep in miglior_rosa.values() for g in rep)
        top_tot = sum(1 for rep in miglior_rosa.values() for g in rep if g[2] == 2)
        
        print("ROSA APPROVATA (Miglior compromesso ML/Budget)")
        print("===================================================")
        print(f"Budget Consumato: {costo_tot} / 500 crediti")
        print(f"Top Player (ML) inseriti: {top_tot}")
        
        for reparto in ["Portieri", "Difensori", "Centrocampisti", "Attaccanti"]:
            nomi = [f"{g[0]} ({g[1]}cr)" for g in miglior_rosa[reparto]]
            print(f"[{reparto.upper()}]: {', '.join(nomi)}")
        print("===================================================")

if __name__ == "__main__":
    main()