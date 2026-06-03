import pandas as pd
import numpy as np
import os
from run_analysis import run_full_analysis

def generate_synthetic_data(n_samples=1000, random_seed=42):
    """
    Génère des données synthétiques réalistes pour Orbitel.
    Le churn est corrélé à des réclamations élevées, une faible satisfaction et des factures élevées.
    """
    np.random.seed(random_seed)
    
    # Génération des caractéristiques
    client_ids = [f"CUST_{i:05d}" for i in range(1, n_samples + 1)]
    age = np.random.randint(18, 80, size=n_samples)
    anciennete = np.random.randint(1, 120, size=n_samples) # mois
    facture_mensuelle = np.random.uniform(20.0, 120.0, size=n_samples)
    
    # Complément réclamations : plus fréquent si ancienneté élevée ou facture élevée
    # Plage de 0 à 10 réclamations
    reclamations = np.random.negative_binomial(n=2, p=0.4, size=n_samples)
    reclamations = np.clip(reclamations, 0, 12)
    
    # Satisfaction : fortement inversement proportionnelle aux réclamations
    # Échelle de 1.0 à 5.0
    satisfaction = 5.0 - (reclamations * 0.4) - np.random.uniform(0, 1.0, size=n_samples)
    satisfaction = np.clip(satisfaction, 1.0, 5.0)
    
    # Logit pour la probabilité de Churn
    # Le churn augmente avec : reclamations, facture_mensuelle, et diminue avec l'age et la satisfaction
    logit = (
        -2.5 
        + 0.5 * reclamations 
        - 1.2 * (satisfaction - 3) 
        + 0.02 * facture_mensuelle 
        - 0.01 * age
    )
    
    # Probabilité sigmoid
    prob_churn = 1 / (1 + np.exp(-logit))
    
    # Variable cible
    churn_y = np.random.binomial(1, prob_churn)
    
    df = pd.DataFrame({
        'ClientID': client_ids,
        'Age': age,
        'Anciennete': anciennete,
        'FactureMensuelle': facture_mensuelle,
        'reclamations': reclamations,
        'satisfaction': satisfaction,
        'Churn_Y': churn_y
    })
    
    return df

def main():
    print("=" * 60)
    print(" GÉNÉRATION DE DONNÉES SYNTHÉTIQUES DE TEST ")
    print("=" * 60)
    
    # 1. Génération des jeux de données
    print("Génération de churn_train.csv (1000 clients)...")
    df_train = generate_synthetic_data(n_samples=1000, random_seed=42)
    df_train.to_csv('churn_train.csv', index=False, sep=';')
    
    print("Génération de churn_test.csv (300 clients)...")
    df_test = generate_synthetic_data(n_samples=300, random_seed=100)
    df_test.to_csv('churn_test.csv', index=False, sep=';')
    
    print("Génération de churn_new.csv (150 nouveaux clients sans cible)...")
    df_new = generate_synthetic_data(n_samples=150, random_seed=999)
    # Supprimer la cible pour simuler les nouveaux clients
    df_new = df_new.drop(columns=['Churn_Y'])
    df_new.to_csv('churn_new.csv', index=False, sep=';')
    
    print("\nFichiers créés avec succès :")
    print("  - churn_train.csv")
    print("  - churn_test.csv")
    print("  - churn_new.csv")
    
    # 2. Exécution du pipeline
    print("\nLancement du pipeline d'analyse pour vérification...")
    run_full_analysis()
    
    print("\n" + "=" * 60)
    print(" NOTE À L'ATTENTION DE L'UTILISATEUR ")
    print("=" * 60)
    print("Le code a été exécuté avec succès sur ces données factices.")
    print("Lorsque vous disposerez de vos vraies données Orbitel, il vous suffira")
    print("de remplacer les fichiers 'churn_train.csv', 'churn_test.csv' et")
    print("'churn_new.csv' par les vôtres, puis de lancer la commande suivante :")
    print("   python run_analysis.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
