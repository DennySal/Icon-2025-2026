import pandas as pd
import time
from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import os

# Configurazione endpoint DBpedia
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

def estrai_dati_dbpedia(player_uri):
    """
    Estrae Età, Trofei e Nazionalità da DBpedia.
    Implementa la gestione della Open World Assumption.
    """
    # Valori di Fallback/Default in caso di Data Sparsity
    risultati = {
        "Eta": 25,       # Età mediana se non trovata
        "Trofei": 0,     # Se non ha la proprietà award, assumiamo 0 trofei
        "Straniero": 1   # Assumiamo straniero di default
    }
    
    # Query SPARQL con blocchi OPTIONAL per evitare fallimenti se manca un dato
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    
    SELECT ?birthDate (COUNT(?award) AS ?num_awards) ?nationality WHERE {{
      OPTIONAL {{ <http://dbpedia.org/resource/{player_uri}> dbo:birthDate ?birthDate . }}
      OPTIONAL {{ <http://dbpedia.org/resource/{player_uri}> dbo:award ?award . }}
      OPTIONAL {{ <http://dbpedia.org/resource/{player_uri}> dbo:nationality ?nationality . }}
    }} GROUP BY ?birthDate ?nationality LIMIT 1
    """
    
    try:
        sparql.setQuery(query)
        res = sparql.query().convert()
        bindings = res["results"]["bindings"]
        
        if bindings:
            dati = bindings[0]
            
            # 1. Calcolo Età dalla Data di Nascita
            if "birthDate" in dati:
                birth_str = dati["birthDate"]["value"][:10] # Prende YYYY-MM-DD
                try:
                    birth_date = datetime.strptime(birth_str, "%Y-%m-%d")
                    risultati["Eta"] = datetime.now().year - birth_date.year
                except ValueError:
                    pass
            
            # 2. Conteggio Trofei/Premi
            if "num_awards" in dati:
                risultati["Trofei"] = int(dati["num_awards"]["value"])
                
            # 3. Verifica Nazionalità (Italiano = 0, Straniero = 1)
            if "nationality" in dati:
                nat = dati["nationality"]["value"].lower()
                if "italy" in nat or "italian" in nat or "italia" in nat:
                    risultati["Straniero"] = 0
                    
    except Exception as e:
        print(f"Errore durante l'estrazione per {player_uri}: {e}")
        
    return risultati

def main():
    print("==================================================")
    print("FASE 1: ESTRAZIONE SEMANTICA (DBPEDIA)")
    print("==================================================")
    
    # 1. Creazione di un dataset fittizio di base per il test
    dati_base = {
        "ID": [1, 2, 3, 4, 5],
        "Nome": ["Lautaro Martinez", "Nicolò Barella", "Khvicha Kvaratskhelia", "Mike Maignan", "Lorenzo Pirola"],
        "URI_Nome": ["Lautaro_Martínez", "Nicolò_Barella", "Khvicha_Kvaratskhelia", "Mike_Maignan", "Lorenzo_Pirola"],
        "Ruolo": ["A", "C", "A", "P", "D"],
        "Quotazione": [40, 22, 35, 18, 4],
        "Media_Voto": [6.50, 6.30, 6.45, 6.20, 5.80],
        "Gol": [24, 2, 11, 0, 0],
        "Assist": [3, 6, 6, 0, 0]
    }
    df = pd.DataFrame(dati_base)
    print("[1/3] Dataset locale caricato con successo.")
    
    # 2. Arricchimento semantico tramite SPARQL
    print("[2/3] Interrogazione di DBpedia in corso...")
    eta_list, trofei_list, straniero_list = [], [], []
    
    # Uso tqdm per una barra di caricamento professionale
    for uri in tqdm(df["URI_Nome"], desc="Scouting Semantico"):
        dati_estratti = estrai_dati_dbpedia(uri)
        eta_list.append(dati_estratti["Eta"])
        trofei_list.append(dati_estratti["Trofei"])
        straniero_list.append(dati_estratti["Straniero"])
        time.sleep(0.5) # Pausa per non sovraccaricare il server DBpedia
        
    df["Eta"] = eta_list
    df["Trofei_Vinti"] = trofei_list
    df["Straniero"] = straniero_list
    
    # 3. Salvataggio del dataset pronto per il Machine Learning
    os.makedirs("data", exist_ok=True)
    percorso_salvataggio = "data/giocatori_arricchiti.csv"
    df.to_csv(percorso_salvataggio, index=False)
    
    print(f"\n[3/3] Estrazione completata! File '{percorso_salvataggio}' salvato.")
    print("\nEcco un'anteprima delle nuove feature semantiche estratte:")
    print(df[["Nome", "Eta", "Trofei_Vinti", "Straniero"]])
    print("==================================================")

if __name__ == "__main__":
    main()