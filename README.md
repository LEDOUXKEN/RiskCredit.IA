# 🏦 Système de Détection du Risque de Crédit

## 📋 Description
Application Streamlit d'analyse prédictive du risque de défaut de paiement utilisant l'Intelligence Artificielle.

**Développé par :** Fidèle Ledoux  
**Formation :** IA School - Intelligence Artificielle & Data Science  

## 🚀 Fonctionnalités

### 🤖 Analyse IA
- Modèle Decision Tree avec 88.9% de précision
- Preprocessing automatique des données
- Prédiction en temps réel du risque de défaut

### 💰 Simulateur de Remboursement
- Calcul des mensualités
- Tableau d'amortissement
- Coût total et intérêts
- Durée personnalisable (1-30 ans)

### 📊 Visualisations
- Score de risque interactif
- Analyse des facteurs de risque
- Graphiques Plotly dynamiques

### 💡 Interface Intuitive
- Motifs de crédit avec emojis
- Métriques en temps réel
- Design bancaire professionnel

## 🛠️ Technologies Utilisées
- **Streamlit** - Framework web
- **Scikit-learn** - Machine Learning
- **Plotly** - Visualisations interactives
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques

## 📊 Performance du Modèle
- **Précision :** 88.9%
- **Rappel :** 76.8%
- **Score F1 :** 75.4%
- **AUC-ROC :** 84.5%

## 📁 Structure du Projet
```
├── apps.py                    # Application Streamlit principale
├── requirements.txt           # Dépendances Python
├── tree_model.pkl            # Modèle ML entraîné
├── credit_risk_dataset.csv   # Dataset d'entraînement
├── Prediction.ipynb          # Notebook d'analyse
└── README.md                 # Documentation
```

## 🚀 Déploiement

### En local :
```bash
pip install -r requirements.txt
streamlit run apps.py
```

### Sur Streamlit Cloud :
1. Fork ce repository
2. Connectez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. Déployez avec `apps.py` comme fichier principal

## 📈 Dataset
- **Source :** Données de crédit synthétiques
- **Taille :** 32,583 observations
- **Variables :** 11 prédicteurs principaux
- **Cible :** Risque de défaut binaire

## 🎯 Variables Prédictives
- Âge du demandeur
- Revenu annuel
- Statut de propriété
- Ancienneté emploi
- Motif du prêt
- Grade du prêt
- Montant et taux d'intérêt
- Historique de crédit

## 📞 Contact
**Fidèle Ledoux**  
Étudiant IA School  
Spécialisation : Data Science & Machine Learning

---
🏅 **Projet Académique Premium** - Solution Bancaire Intelligente
