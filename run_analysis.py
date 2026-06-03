import os
import pandas as pd
import numpy as np
from src.model import (
    load_data,
    add_frustration_feature,
    train_baseline_model,
    train_improved_model,
    evaluate_model,
    optimize_threshold
)
from sklearn.impute import SimpleImputer

def prepare_features(df_train: pd.DataFrame, df_test: pd.DataFrame, df_new: pd.DataFrame = None):
    """
    Sépare les variables explicatives et la cible.
    Gère l'imputation des valeurs manquantes et élimine les identifiants si présents.
    """
    # Identifier la cible
    target_col = 'Churn_Y'
    if target_col not in df_train.columns:
        # Recherche insensible à la casse
        for col in df_train.columns:
            if col.lower() == 'churn_y' or col.lower() == 'churn':
                target_col = col
                break
                
    # Séparation X et y
    y_train = df_train[target_col] if target_col in df_train.columns else None
    y_test = df_test[target_col] if target_col in df_test.columns else None
    
    # Drop de la cible et des identifiants potentiels
    cols_to_drop = [target_col] if target_col in df_train.columns else []
    id_keywords = ['id', 'client', 'customer', 'uuid']
    for col in df_train.columns:
        if any(kw in col.lower() for kw in id_keywords) and col != target_col:
            cols_to_drop.append(col)
            
    X_train = df_train.drop(columns=cols_to_drop, errors='ignore')
    X_test = df_test.drop(columns=cols_to_drop, errors='ignore')
    
    X_new = None
    if df_new is not None:
        X_new = df_new.drop(columns=cols_to_drop, errors='ignore')
        
    # S'assurer qu'on ne garde que les colonnes numériques pour ce modèle de base
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    if X_new is not None:
        X_new = X_new.select_dtypes(include=[np.number])
        
    # Imputation des valeurs manquantes si nécessaire (avec la médiane)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    X_new_imp = None
    if X_new is not None:
        X_new_imp = pd.DataFrame(imputer.transform(X_new), columns=X_new.columns)
        
    return X_train_imp, y_train, X_test_imp, y_test, X_new_imp

def run_full_analysis():
    print("=" * 60)
    print(" PIPELINE DE DÉCISION MÉTIER - CHURN ORBITEL")
    print("=" * 60)
    
    # Vérification de l'existence des fichiers
    train_path = 'churn_train.csv'
    test_path = 'churn_test.csv'
    new_path = 'churn_new.csv'
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"\n[ERREUR] Les fichiers de données '{train_path}' et/ou '{test_path}' sont introuvables.")
        print("Veuillez les copier dans ce répertoire ou exécuter 'python test_pipeline.py' pour générer des données factices et tester le code.")
        return
        
    # 1. Chargement des données
    print("\n--- 1. CHARGEMENT DES DONNÉES ---")
    df_train = load_data(train_path)
    df_test = load_data(test_path)
    
    df_new = None
    if os.path.exists(new_path):
        df_new = load_data(new_path)
    else:
        print(f"Note : '{new_path}' non détecté. Les prédictions finales ne seront pas générées.")
        
    # --- PARTIE 1 : ANALYSE DU MODÈLE INITIAL ---
    print("\n--- PARTIE 1 : ANALYSE DU MODÈLE INITIAL (BASELINE) ---")
    X_train, y_train, X_test, y_test, X_new = prepare_features(df_train, df_test, df_new)
    
    print("\nEntraînement de la régression logistique initiale...")
    baseline_model = train_baseline_model(X_train, y_train)
    
    # Évaluation au seuil par défaut 0.5
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, threshold=0.5)
    
    print("\nRésultats du modèle initial (Seuil 0.5) :")
    print(f"  - Accuracy  : {baseline_metrics['accuracy']:.4f}")
    print(f"  - Précision : {baseline_metrics['precision']:.4f}")
    print(f"  - Recall    : {baseline_metrics['recall']:.4f}")
    print(f"  - F1-Score  : {baseline_metrics['f1_score']:.4f}")
    print("\nMatrice de Confusion :")
    print(f"  - Vrais Négatifs (VN) : {baseline_metrics['TN']} | Faux Positifs (FP) : {baseline_metrics['FP']}")
    print(f"  - Faux Négatifs (FN) : {baseline_metrics['FN']} | Vrais Positifs (VP) : {baseline_metrics['TP']}")
    print(f"\nCoût Métier Associé :")
    print(f"  - Coût Faux Négatifs (FN * 200€) : {baseline_metrics['fn_cost']:.2f}€")
    print(f"  - Coût Faux Positifs (FP * 50€)  : {baseline_metrics['fp_cost']:.2f}€")
    print(f"  - COÛT MÉTIER TOTAL             : {baseline_metrics['total_cost']:.2f}€")
    
    # --- PARTIE 2 : AMÉLIORATION DU MODÈLE ---
    print("\n--- PARTIE 2 : AMÉLIORATION DU MODÈLE (SCALING & CLASS WEIGHT) ---")
    print("Entraînement du modèle amélioré (Standardisation + Équilibrage des classes + max_iter=1000)...")
    improved_model = train_improved_model(X_train, y_train, class_weight='balanced')
    
    improved_metrics = evaluate_model(improved_model, X_test, y_test, threshold=0.5)
    print("\nRésultats du modèle amélioré (Seuil 0.5) :")
    print(f"  - Accuracy  : {improved_metrics['accuracy']:.4f}")
    print(f"  - Précision : {improved_metrics['precision']:.4f}")
    print(f"  - Recall    : {improved_metrics['recall']:.4f}")
    print(f"  - F1-Score  : {improved_metrics['f1_score']:.4f}")
    print(f"  - COÛT MÉTIER TOTAL             : {improved_metrics['total_cost']:.2f}€")
    print(f"  - Réduction de coût par rapport à la baseline : {baseline_metrics['total_cost'] - improved_metrics['total_cost']:.2f}€")
    
    # --- PARTIE 3 : OPTIMISATION MÉTIER (VARIABLE FRUSTRATION ET SEUIL) ---
    print("\n--- PARTIE 3 : OPTIMISATION MÉTIER & VARIABLE FRUSTRATION ---")
    print("1. Création de la variable 'frustration' sur les jeux de données...")
    df_train_frust = add_frustration_feature(df_train)
    df_test_frust = add_frustration_feature(df_test)
    
    df_new_frust = None
    if df_new is not None:
        df_new_frust = add_frustration_feature(df_new)
        
    X_train_frust, y_train, X_test_frust, y_test, X_new_frust = prepare_features(df_train_frust, df_test_frust, df_new_frust)
    
    print("\nEntraînement du modèle final avec la variable 'frustration' intégrée...")
    final_model = train_improved_model(X_train_frust, y_train, class_weight='balanced')
    
    print("\n2. Optimisation du seuil de décision (Calcul pour les seuils 0.1 à 0.9)...")
    best_threshold, best_metrics, df_thresholds = optimize_threshold(final_model, X_test_frust, y_test)
    
    # Affichage du tableau comparatif des seuils
    print("\nTableau de performance par Seuil :")
    print(df_thresholds[['threshold', 'accuracy', 'precision', 'recall', 'FN', 'FP', 'total_cost']].to_string(index=False))
    
    print("\n" + "-" * 50)
    print(" SEUIL OPTIMAL SÉLECTIONNÉ ")
    print("-" * 50)
    print(f"Seuil de décision optimal : {best_threshold:.1f}")
    print(f"Coût métier associé       : {best_metrics['total_cost']:.2f}€")
    print(f"Structure des erreurs     : FN={best_metrics['FN']} ({best_metrics['fn_cost']:.0f}€) | FP={best_metrics['FP']} ({best_metrics['fp_cost']:.0f}€)")
    print(f"Accuracy                  : {best_metrics['accuracy']:.4f}")
    print(f"Recall (Sensibilité)      : {best_metrics['recall']:.4f}")
    print(f"Précision                 : {best_metrics['precision']:.4f}")
    print("-" * 50)
    
    # 4. Prédiction sur le jeu final churn_new.csv
    if X_new_frust is not None:
        print("\n--- 4. GÉNÉRATION DES PRÉDICTIONS SUR CHURN_NEW.CSV ---")
        # Obtenir les probabilités d'appartenance à la classe Churn (1)
        probs_new = final_model.predict_proba(X_new_frust)[:, 1]
        # Appliquer le seuil de décision optimal
        preds_new = (probs_new >= best_threshold).astype(int)
        
        # Enregistrer dans un DataFrame
        df_predictions = pd.DataFrame({
            'Probabilite_Churn': probs_new,
            'Pred_Churn_Y': preds_new
        })
        
        # Si le fichier d'origine avait un identifiant client, le conserver
        id_cols = [col for col in df_new.columns if any(kw in col.lower() for kw in ['id', 'client', 'customer'])]
        if id_cols:
            df_predictions.insert(0, id_cols[0], df_new[id_cols[0]])
            
        output_file = 'predictions_finales.csv'
        df_predictions.to_csv(output_file, index=False, sep=';')
        print(f"Prédictions enregistrées avec succès dans : '{output_file}'")
        print(f"Nombre de clients qui vont churner selon le modèle : {sum(preds_new)} sur {len(preds_new)} clients.")
    
    print("\nAnalyse terminée avec succès !")
    print("=" * 60)

if __name__ == "__main__":
    run_full_analysis()
