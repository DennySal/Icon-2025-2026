import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def genera_dataset_sintetico():
    """Genera un dataset realistico di 150 giocatori per il training."""
    np.random.seed(42)
    n_samples = 150
    
    ruoli = np.random.choice(['P', 'D', 'C', 'A'], n_samples)
    quotazioni = np.random.randint(1, 45, n_samples)
    medie_voto = np.random.uniform(5.0, 7.5, n_samples)
    gol = np.random.randint(0, 25, n_samples)
    eta = np.random.randint(18, 38, n_samples)
    straniero = np.random.choice([0, 1], n_samples)
    trofei = np.random.randint(0, 10, n_samples)
    
    # Assegnazione logica della Classe Target (0: Flop, 1: Titolare, 2: Top Player)
    target = []
    for i in range(n_samples):
        if medie_voto[i] > 6.5 and gol[i] > 8:
            target.append(2) # Top Player
        elif medie_voto[i] >= 6.0:
            target.append(1) # Titolare
        else:
            target.append(0) # Flop
            
    df = pd.DataFrame({
        'Ruolo': ruoli, 'Quotazione': quotazioni, 'Media_Voto': medie_voto,
        'Gol': gol, 'Eta': eta, 'Straniero': straniero, 'Trofei_Vinti': trofei,
        'Target_Class': target
    })
    return df

def plot_learning_curve(estimator, X, y):
    """Genera e salva il grafico delle Learning Curves"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1_macro'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score (F1)")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-Validation Score (F1)")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")
    
    plt.title("Curve di Apprendimento (Random Forest + SMOTE)")
    plt.xlabel("Numero di Esempi di Training")
    plt.ylabel("F1-Score (Macro)")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    os.makedirs("../test", exist_ok=True)
    plt.savefig("../test/learning_curves.png")
    print("\n[!] Grafico Learning Curves salvato in '../test/learning_curves.png'")

def main():
    print("==================================================")
    print("FASE 2: MACHINE LEARNING & SCOUTING PREDITTIVO")
    print("==================================================")
    
    df = genera_dataset_sintetico()
    X = df.drop(columns=['Target_Class'])
    y = df['Target_Class']
    
    print(f"Dataset caricato: {len(df)} giocatori.")
    print(f"Distribuzione Classi Originale: \n{y.value_counts().to_string()}\n")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Quotazione', 'Media_Voto', 'Gol', 'Eta', 'Trofei_Vinti']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Ruolo', 'Straniero'])
        ])
    
    # FIX DEFINITIVO: k_neighbors=1 per evitare crash nei micro-fold delle learning curves
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(k_neighbors=1, random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=2, random_state=42))
    ])
    
    print("Addestramento e 10-Fold Cross-Validation in corso...\n")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    print("=== RISULTATI MEDI (10-FOLD CV) ===")
    print(f"Accuracy:        {np.mean(results['test_accuracy']):.3f} (+/- {np.std(results['test_accuracy']):.3f})")
    print(f"Precision Macro: {np.mean(results['test_precision_macro']):.3f} (+/- {np.std(results['test_precision_macro']):.3f})")
    print(f"Recall Macro:    {np.mean(results['test_recall_macro']):.3f} (+/- {np.std(results['test_recall_macro']):.3f})")
    print(f"F1-Score Macro:  {np.mean(results['test_f1_macro']):.3f} (+/- {np.std(results['test_f1_macro']):.3f})")
    print("===================================")
    
    plot_learning_curve(pipeline, X, y)
    
    pipeline.fit(X, y)
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/ai_scout_classifier.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Modello serializzato e salvato in '{model_path}'")
    print("Pronto per essere usato dal risolutore CSP!")
    print("==================================================")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()