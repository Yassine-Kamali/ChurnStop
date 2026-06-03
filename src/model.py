import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings

def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV de manière robuste en testant les séparateurs usuels (, et ;).
    """
    try:
        # Essayer d'abord avec le séparateur par défaut (virgule)
        df = pd.read_csv(file_path, sep=',')
        # Si une seule colonne est détectée, essayer avec un point-virgule
        if df.shape[1] <= 1:
            df = pd.read_csv(file_path, sep=';')
        print(f"Chargement réussi : {file_path} (Taille : {df.shape[0]} lignes, {df.shape[1]} colonnes)")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Erreur lors du chargement de {file_path} : {str(e)}")

def detect_columns(df: pd.DataFrame):
    """
    Détecte dynamiquement les colonnes de réclamations et de satisfaction dans le DataFrame.
    """
    cols = df.columns.tolist()
    
    # Recherche pour les réclamations
    reclam_keywords = ['reclamation', 'complaint', 'claim', 'nb_rec', 'num_rec', 'reclamations']
    reclam_col = None
    for kw in reclam_keywords:
        for col in cols:
            if kw in col.lower():
                reclam_col = col
                break
        if reclam_col:
            break
            
    # Recherche pour la satisfaction
    satisf_keywords = ['satisfaction', 'satisf', 'score_sat', 'sat_level']
    satisf_col = None
    for kw in satisf_keywords:
        for col in cols:
            if kw in col.lower():
                satisf_col = col
                break
        if satisf_col:
            break
            
    # Valeurs par défaut si non trouvées
    if not reclam_col:
        reclam_col = 'reclamations'
        print(f"Attention : Colonne de réclamations non trouvée. Utilisation de la valeur par défaut : '{reclam_col}'")
    else:
        print(f"Colonne de réclamations détectée : '{reclam_col}'")
        
    if not satisf_col:
        satisf_col = 'satisfaction'
        print(f"Attention : Colonne de satisfaction non trouvée. Utilisation de la valeur par défaut : '{satisf_col}'")
    else:
        print(f"Colonne de satisfaction détectée : '{satisf_col}'")
        
    return reclam_col, satisf_col

def add_frustration_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable de frustration (0 ou 1) selon la règle métier :
    Un client est frustré si :
    - nombre de réclamations > 5
    ET
    - niveau de satisfaction < 2.5
    """
    df_copy = df.copy()
    reclam_col, satisf_col = detect_columns(df_copy)
    
    if reclam_col in df_copy.columns and satisf_col in df_copy.columns:
        # Application de la règle métier stricte
        df_copy['frustration'] = np.where(
            (df_copy[reclam_col] > 5) & (df_copy[satisf_col] < 2.5), 
            1, 
            0
        )
        print("Variable 'frustration' créée avec succès.")
    else:
        # Si les colonnes n'existent pas du tout dans les données fournies, on crée une colonne de zéros
        df_copy['frustration'] = 0
        print("Attention : Colonnes requises pour la frustration absentes. Variable 'frustration' initialisée à 0.")
        
    return df_copy

def calculate_business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcule le coût métier total selon les coefficients fournis :
    - Faux Négatif (FN) : Le client va churner (1) mais le modèle prédit fidèle (0) -> Coût = 200€
    - Faux Positif (FP) : Le client est fidèle (0) mais le modèle prédit churner (1) -> Coût = 50€
    - Vrai Négatif (VN) & Vrai Positif (VP) : Prédictions correctes -> Coût = 0€
    
    Formule : Coût = 200 * FN + 50 * FP
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = 200 * fn + 50 * fp
    return {
        'total_cost': float(cost),
        'fn_cost': float(200 * fn),
        'fp_cost': float(50 * fp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Entraîne le modèle de base (régression logistique) avec les paramètres par défaut.
    Peut lever un avertissement de non-convergence si les données ne sont pas standardisées.
    """
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_improved_model(X_train: pd.DataFrame, y_train: pd.Series, class_weight='balanced') -> Pipeline:
    """
    Entraîne un modèle amélioré incluant :
    1. Mise à l'échelle des données (StandardScaler) pour accélérer la convergence et améliorer les performances.
    2. Augmentation de max_iter à 1000 pour éliminer les warnings de convergence.
    3. Option de pondération des classes (class_weight='balanced') pour gérer le déséquilibre.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X: pd.DataFrame, y_true: pd.Series, threshold: float = 0.5) -> dict:
    """
    Évalue le modèle pour un seuil de décision donné.
    Si threshold=0.5, utilise la méthode predict par défaut.
    Sinon, utilise predict_proba pour appliquer le seuil personnalisé.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        
    # Calcul des métriques standard
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calcul des coûts métier
    cost_metrics = calculate_business_cost(y_true.values if hasattr(y_true, 'values') else y_true, y_pred)
    
    metrics = {
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        **cost_metrics
    }
    return metrics

def optimize_threshold(model, X: pd.DataFrame, y_true: pd.Series) -> tuple:
    """
    Calcule le coût métier pour des seuils de décision allant de 0.1 à 0.9.
    Identifie le seuil optimal (celui qui minimise le coût total).
    
    Retourne :
    - Le seuil optimal (float)
    - Le coût minimal associé (dict)
    - Un DataFrame contenant les résultats détaillés pour chaque seuil.
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    results = []
    
    for t in thresholds:
        metrics = evaluate_model(model, X, y_true, threshold=t)
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    
    # Trouver l'index du coût total minimal
    best_idx = df_results['total_cost'].idxmin()
    best_threshold = df_results.loc[best_idx, 'threshold']
    best_metrics = df_results.loc[best_idx].to_dict()
    
    return best_threshold, best_metrics, df_results
