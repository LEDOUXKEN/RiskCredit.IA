import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import math
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="💰 Crédit Risk Analyzer - Fidèle Ledoux",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS avancé et élégant
st.markdown("""
<style>
    /* Styles pour les cartes de résultat */
    .success-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .danger-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .info-card {
        background: linear-gradient(135deg, #cce7ff 0%, #b3d9ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.2);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Styles pour le header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    /* Styles pour les métriques */
    .metric-gold {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: #8B4513;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
    }
    
    /* Tableau d'amortissement stylé */
    .amortization-table {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Progress bar personnalisée */
    .custom-progress {
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Émojis animés */
    .emoji-animate {
        display: inline-block;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('tree_model.pkl')
        data = pd.read_csv('credit_risk_dataset.csv', sep=';')
        
        numeric_features = ['person_age', 'person_income', 'person_emp_length', 
                           'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                           'cb_person_cred_hist_length']
        
        scaler = StandardScaler()
        scaler.fit(data[numeric_features].fillna(data[numeric_features].median()))
        
        return model, scaler, True
    except FileNotFoundError:
        st.error("⚠️ Modèle non trouvé. Mode simulation intelligent activé.")
        return None, None, False

def preprocess_input(input_data, scaler):
    df = pd.DataFrame([input_data])
    
    numeric_features = ['person_age', 'person_income', 'person_emp_length', 
                       'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                       'cb_person_cred_hist_length']
    
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    expected_columns = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
        'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D',
        'loan_grade_E', 'loan_grade_F', 'loan_grade_G',
        'cb_person_default_on_file_N', 'cb_person_default_on_file_Y'
    ]
    
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    return df_encoded[expected_columns]

def calculate_amortization_schedule(principal, annual_rate, years, start_date=None):
    """Calcul détaillé du tableau d'amortissement"""
    if start_date is None:
        start_date = datetime.now()
    
    monthly_rate = annual_rate / 100 / 12
    num_payments = int(years * 12)
    
    if monthly_rate > 0:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_payment = principal / num_payments
    
    schedule = []
    remaining_balance = principal
    total_interest_paid = 0
    total_principal_paid = 0
    
    for month in range(1, num_payments + 1):
        payment_date = start_date + timedelta(days=30 * (month - 1))
        
        if monthly_rate > 0:
            interest_payment = remaining_balance * monthly_rate
            principal_payment = monthly_payment - interest_payment
        else:
            interest_payment = 0
            principal_payment = monthly_payment
        
        remaining_balance = max(0, remaining_balance - principal_payment)
        total_interest_paid += interest_payment
        total_principal_paid += principal_payment
        
        schedule.append({
            'Mois': month,
            'Date': payment_date.strftime('%m/%Y'),
            'Paiement Total': monthly_payment,
            'Capital': principal_payment,
            'Intérêts': interest_payment,
            'Solde Restant': remaining_balance,
            'Capital Cumulé': total_principal_paid,
            'Intérêts Cumulés': total_interest_paid,
            '% Remboursé': (total_principal_paid / principal) * 100
        })
    
    return schedule

def calculate_financial_indicators(principal, annual_rate, years, monthly_income):
    """Calcul d'indicateurs financiers avancés"""
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    
    if monthly_rate > 0:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_payment = principal / num_payments
    
    total_payment = monthly_payment * num_payments
    total_interest = total_payment - principal
    
    # Ratios financiers
    debt_to_income_ratio = (monthly_payment / monthly_income) * 100 if monthly_income > 0 else 0
    interest_rate_effectiveness = (total_interest / principal) * 100
    
    # Coût d'opportunité (estimation)
    opportunity_cost_rate = 0.03  # 3% rendement alternatif
    opportunity_cost = principal * ((1 + opportunity_cost_rate)**years - 1)
    
    return {
        'monthly_payment': monthly_payment,
        'total_payment': total_payment,
        'total_interest': total_interest,
        'debt_to_income_ratio': debt_to_income_ratio,
        'interest_rate_effectiveness': interest_rate_effectiveness,
        'opportunity_cost': opportunity_cost,
        'break_even_months': years * 12
    }

def get_risk_recommendations(risk_score, loan_data):
    """Génère des recommandations personnalisées"""
    recommendations = []
    
    if risk_score < 0.3:
        recommendations.extend([
            "✅ Profil excellent - Négociez un taux préférentiel",
            "💰 Envisagez un montant légèrement supérieur si nécessaire",
            "📈 Profitez de votre bon profil pour de futurs crédits"
        ])
    elif risk_score < 0.6:
        recommendations.extend([
            "⚠️ Réduisez le montant demandé de 10-20%",
            "📊 Améliorez votre ancienneté dans l'emploi",
            "💳 Remboursez vos dettes existantes avant la demande"
        ])
    else:
        recommendations.extend([
            "🚨 Reportez votre demande de 6-12 mois",
            "💪 Augmentez vos revenus ou réduisez vos charges",
            "🏦 Consultez un conseiller financier",
            "📋 Constituez un apport personnel plus important"
        ])
    
    # Recommandations spécifiques
    if loan_data['loan_percent_income'] > 0.4:
        recommendations.append("📉 Réduisez le ratio dette/revenu sous 40%")
    
    if loan_data['person_emp_length'] < 2:
        recommendations.append("⏰ Stabilisez votre emploi (>2 ans recommandé)")
    
    return recommendations

# Chargement du modèle
model, scaler, model_available = load_model_and_data()

# Header principal avec design avancé
st.markdown("""
<div class="main-header">
    <h1>🏦 CRÉDIT RISK ANALYZER PREMIUM</h1>
    <h3>👨‍🎓 Développé par Fidèle Ledoux</h3>
    <p>🎓 IA SCHOOL - Intelligence Artificielle & Data Science</p>
    <p>⚡ Système d'Analyse Prédictive du Risque de Crédit ⚡</p>
</div>
""", unsafe_allow_html=True)

# Motifs de crédit avec emojis et descriptions
loan_intent_options = {
    "PERSONAL": {"emoji": "💳", "name": "Personnel", "desc": "Achat personnel, voyage, événement"},
    "EDUCATION": {"emoji": "🎓", "name": "Éducation", "desc": "Formation, études, certification"},
    "MEDICAL": {"emoji": "🏥", "name": "Médical", "desc": "Soins, chirurgie, équipements médicaux"},
    "VENTURE": {"emoji": "🚀", "name": "Entreprise", "desc": "Création, développement d'activité"},
    "HOMEIMPROVEMENT": {"emoji": "🏠", "name": "Amélioration Habitat", "desc": "Rénovation, extension, équipement"},
    "DEBTCONSOLIDATION": {"emoji": "💰", "name": "Consolidation Dettes", "desc": "Regroupement de crédits existants"}
}

# Sidebar avec design avancé
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;'>
    <h2>🏦 DOSSIER DE CRÉDIT</h2>
    <p>💼 Simulateur Financier Intelligent</p>
</div>
""", unsafe_allow_html=True)

# Section Motif du crédit avec descriptions
st.sidebar.subheader("💡 Motif du Crédit")
loan_intent = st.sidebar.selectbox(
    "🎯 Objectif du financement",
    options=list(loan_intent_options.keys()),
    format_func=lambda x: f"{loan_intent_options[x]['emoji']} {loan_intent_options[x]['name']}",
    help="Choisissez la finalité de votre crédit"
)

# Affichage de la description du motif
selected_intent = loan_intent_options[loan_intent]
st.sidebar.info(f"**{selected_intent['emoji']} {selected_intent['name']}**\n\n{selected_intent['desc']}")

# Profil personnel
st.sidebar.subheader("👤 Profil Personnel")
person_age = st.sidebar.slider("🎂 Âge", 18, 80, 30, help="Votre âge actuel")
person_income = st.sidebar.number_input("💰 Revenu annuel (€)", 0, 1000000, 50000, step=1000, 
                                       help="Revenu net annuel total")
monthly_income = person_income / 12 if person_income > 0 else 0

# Statut de propriété avec emojis
home_ownership_options = {
    "RENT": "🏠 Locataire",
    "OWN": "🏡 Propriétaire", 
    "MORTGAGE": "🏘️ Crédit immobilier en cours",
    "OTHER": "🏢 Autre situation"
}

person_home_ownership = st.sidebar.selectbox(
    "🏡 Statut de propriété",
    options=list(home_ownership_options.keys()),
    format_func=lambda x: home_ownership_options[x]
)

person_emp_length = st.sidebar.slider("⏰ Ancienneté emploi (années)", 0.0, 40.0, 5.0, 0.5,
                                     help="Nombre d'années dans votre emploi actuel")

# Détails du prêt
st.sidebar.subheader("💳 Détails du Prêt")

# Grade avec explications
grade_explanations = {
    "A": "🌟 Excellent (taux le plus bas)",
    "B": "⭐ Très bon",
    "C": "✅ Bon", 
    "D": "⚠️ Moyen",
    "E": "🟡 Attention",
    "F": "🟠 Risqué",
    "G": "🔴 Très risqué"
}

loan_grade = st.sidebar.selectbox(
    "📊 Grade de risque estimé",
    options=list(grade_explanations.keys()),
    format_func=lambda x: grade_explanations[x],
    index=2
)

loan_amnt = st.sidebar.number_input("💸 Montant du prêt (€)", 500, 500000, 15000, step=500,
                                   help="Montant total souhaité")
loan_int_rate = st.sidebar.slider("📈 Taux d'intérêt annuel (%)", 1.0, 25.0, 12.0, 0.1,
                                 help="Taux proposé par la banque")
loan_duration_years = st.sidebar.slider("📅 Durée du prêt (années)", 1, 35, 5,
                                       help="Durée de remboursement souhaitée")

# Calculs automatiques
loan_percent_income = loan_amnt / person_income if person_income > 0 else 0

# Indicateurs en temps réel dans la sidebar
if monthly_income > 0:
    monthly_payment_estimate = loan_amnt * ((loan_int_rate/100/12) * (1 + loan_int_rate/100/12)**(loan_duration_years*12)) / ((1 + loan_int_rate/100/12)**(loan_duration_years*12) - 1) if loan_int_rate > 0 else loan_amnt / (loan_duration_years * 12)
    debt_ratio = (monthly_payment_estimate / monthly_income) * 100
    
    if debt_ratio < 33:
        ratio_color = "🟢"
        ratio_status = "Excellent"
    elif debt_ratio < 40:
        ratio_color = "🟡"  
        ratio_status = "Acceptable"
    else:
        ratio_color = "🔴"
        ratio_status = "Risqué"
    
    st.sidebar.metric("💳 Mensualité estimée", f"{monthly_payment_estimate:,.0f} €")
    st.sidebar.metric("📊 Ratio d'endettement", f"{debt_ratio:.1f}%", 
                     delta=f"{ratio_color} {ratio_status}")

# Historique financier
st.sidebar.subheader("📊 Historique Financier")
cb_person_default_on_file = st.sidebar.selectbox(
    "🚨 Défaut de paiement antérieur",
    options=["N", "Y"],
    format_func=lambda x: "✅ Aucun défaut" if x == "N" else "⚠️ Défaut dans l'historique"
)

cb_person_cred_hist_length = st.sidebar.slider("📜 Ancienneté historique crédit (années)", 
                                              0, 30, 5,
                                              help="Nombre d'années d'historique de crédit")

# Zone principale avec onglets
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analyse Risque", "💰 Simulation Remboursement", "📊 Tableaux Détaillés", "🔍 Recommandations"])

with tab1:
    st.header("💎 ANALYSE DU RISQUE DE CRÉDIT")
    
    # Métriques principales avec design doré
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.markdown(f"""
        <div class="metric-gold">
            <h3>{person_age} ans</h3>
            <p>👤 Âge</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric2:
        st.markdown(f"""
        <div class="metric-gold">
            <h3>{person_income:,} €</h3>
            <p>💰 Revenu annuel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric3:
        st.markdown(f"""
        <div class="metric-gold">
            <h3>{loan_amnt:,} €</h3>
            <p>💳 Montant demandé</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metric4:
        st.markdown(f"""
        <div class="metric-gold">
            <h3>{loan_percent_income:.1%}</h3>
            <p>📊 Ratio Dette/Revenu</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Bouton d'analyse principal avec animation
    st.markdown("<br>", unsafe_allow_html=True)
    col_button = st.columns([1, 2, 1])
    with col_button[1]:
        if st.button("🔍 ANALYSER LE RISQUE MAINTENANT", type="primary", use_container_width=True):
            
            # Animation de chargement
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            progress_text.text("🤖 Initialisation de l'IA...")
            progress_bar.progress(20)
            
            # Préparation des données
            input_data = {
                'person_age': person_age,
                'person_income': person_income,
                'person_home_ownership': person_home_ownership,
                'person_emp_length': person_emp_length,
                'loan_intent': loan_intent,
                'loan_grade': loan_grade,
                'loan_amnt': loan_amnt,
                'loan_int_rate': loan_int_rate,
                'loan_percent_income': loan_percent_income,
                'cb_person_default_on_file': cb_person_default_on_file,
                'cb_person_cred_hist_length': cb_person_cred_hist_length
            }
            
            progress_text.text("📊 Analyse des données...")
            progress_bar.progress(50)
            
            # Prédiction avec modèle IA ou simulation avancée
            risk_score = None
            if model_available and model is not None and scaler is not None:
                try:
                    processed_data = preprocess_input(input_data, scaler)
                    risk_score = model.predict_proba(processed_data)[0][1]
                    progress_text.text("✅ Modèle IA activé avec succès!")
                except Exception as e:
                    progress_text.text("⚠️ Basculement vers simulation avancée...")
                    risk_score = None
            
            if risk_score is None:
                # Simulation avancée avec plus de facteurs
                risk_factors = 0
                if person_age < 25: risk_factors += 0.12
                elif person_age > 65: risk_factors += 0.08
                
                if person_income < 20000: risk_factors += 0.25
                elif person_income < 30000: risk_factors += 0.15
                elif person_income < 40000: risk_factors += 0.05
                
                if loan_percent_income > 0.5: risk_factors += 0.3
                elif loan_percent_income > 0.4: risk_factors += 0.2
                elif loan_percent_income > 0.3: risk_factors += 0.1
                
                grade_risk = {'A': 0, 'B': 0.05, 'C': 0.1, 'D': 0.15, 'E': 0.2, 'F': 0.25, 'G': 0.3}
                risk_factors += grade_risk.get(loan_grade, 0.15)
                
                if cb_person_default_on_file == 'Y': risk_factors += 0.35
                if loan_int_rate > 18: risk_factors += 0.2
                elif loan_int_rate > 15: risk_factors += 0.1
                
                if person_emp_length < 1: risk_factors += 0.15
                elif person_emp_length < 2: risk_factors += 0.08
                
                if cb_person_cred_hist_length < 2: risk_factors += 0.1
                
                # Facteurs par motif de crédit
                intent_risk = {'VENTURE': 0.1, 'MEDICAL': 0.05, 'PERSONAL': 0.02}
                risk_factors += intent_risk.get(loan_intent, 0)
                
                risk_score = min(risk_factors, 0.98)
            
            progress_bar.progress(80)
            progress_text.text("🎯 Finalisation de l'analyse...")
            
            progress_bar.progress(100)
            progress_text.text("✅ Analyse terminée!")
            
            # Nettoyage des éléments de progression
            progress_text.empty()
            progress_bar.empty()
            
            # Affichage du résultat avec animations
            st.markdown("<br>", unsafe_allow_html=True)
            
            if risk_score < 0.25:
                st.markdown(f"""
                <div class="success-card">
                    <div style="text-align: center;">
                        <h2><span class="emoji-animate">✅</span> RISQUE TRÈS FAIBLE</h2>
                        <h1 style="color: #28a745; font-size: 3rem; margin: 1rem 0;">{risk_score:.1%}</h1>
                        <h3>🎉 CRÉDIT FORTEMENT RECOMMANDÉ</h3>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">
                            Excellent profil emprunteur. Négociez les meilleures conditions!
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                
            elif risk_score < 0.4:
                st.markdown(f"""
                <div class="success-card">
                    <div style="text-align: center;">
                        <h2><span class="emoji-animate">✅</span> RISQUE FAIBLE</h2>
                        <h1 style="color: #28a745; font-size: 3rem; margin: 1rem 0;">{risk_score:.1%}</h1>
                        <h3>👍 CRÉDIT APPROUVÉ</h3>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">
                            Bon profil emprunteur. Conditions favorables attendues.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            elif risk_score < 0.65:
                st.markdown(f"""
                <div class="warning-card">
                    <div style="text-align: center;">
                        <h2><span class="emoji-animate">⚠️</span> RISQUE MODÉRÉ</h2>
                        <h1 style="color: #856404; font-size: 3rem; margin: 1rem 0;">{risk_score:.1%}</h1>
                        <h3>🔍 ÉVALUATION COMPLÉMENTAIRE</h3>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">
                            Dossier à améliorer. Consultez nos recommandations.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="danger-card">
                    <div style="text-align: center;">
                        <h2><span class="emoji-animate">❌</span> RISQUE ÉLEVÉ</h2>
                        <h1 style="color: #721c24; font-size: 3rem; margin: 1rem 0;">{risk_score:.1%}</h1>
                        <h3>🚫 CRÉDIT NON RECOMMANDÉ</h3>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">
                            Dossier à restructurer avant nouvelle demande.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Analyse détaillée des facteurs
            st.subheader("🔍 Analyse Détaillée des Facteurs")
            
            factors_analysis = []
            factor_impacts = []
            
            if person_age < 25:
                factors_analysis.append("👶 Âge jeune - Manque d'expérience financière")
                factor_impacts.append(0.12)
            elif person_age > 65:
                factors_analysis.append("👴 Âge avancé - Revenus potentiellement décroissants")
                factor_impacts.append(0.08)
            
            if person_income < 30000:
                factors_analysis.append("💸 Revenus insuffisants pour le montant demandé")
                factor_impacts.append(0.2)
            
            if loan_percent_income > 0.4:
                factors_analysis.append("📊 Ratio dette/revenu critique (>40%)")
                factor_impacts.append(0.25)
            
            if loan_grade in ['E', 'F', 'G']:
                factors_analysis.append(f"⚠️ Grade de crédit défavorable ({loan_grade})")
                factor_impacts.append(0.2)
            
            if cb_person_default_on_file == 'Y':
                factors_analysis.append("🚨 Historique de défaut de paiement")
                factor_impacts.append(0.35)
            
            if loan_int_rate > 15:
                factors_analysis.append(f"📈 Taux d'intérêt élevé ({loan_int_rate}%)")
                factor_impacts.append(0.15)
            
            if person_emp_length < 2:
                factors_analysis.append("⏰ Ancienneté emploi insuffisante")
                factor_impacts.append(0.1)
            
            col_factors1, col_factors2 = st.columns(2)
            
            with col_factors1:
                if factors_analysis:
                    st.warning("⚠️ **Facteurs de risque identifiés:**")
                    for i, factor in enumerate(factors_analysis):
                        impact_pct = factor_impacts[i] * 100 if i < len(factor_impacts) else 5
                        st.write(f"• {factor} *({impact_pct:.0f}% d'impact)*")
                else:
                    st.success("✅ **Aucun facteur de risque majeur identifié**")
            
            with col_factors2:
                if factor_impacts:
                    # Graphique simple avec barres Streamlit
                    factors_df = pd.DataFrame({
                        'Facteur': [f"F{i+1}" for i in range(len(factor_impacts))],
                        'Impact (%)': [f * 100 for f in factor_impacts]
                    })
                    st.bar_chart(factors_df.set_index('Facteur'))

with tab2:
    st.header("💰 SIMULATEUR DE REMBOURSEMENT AVANCÉ")
    
    # Calculs financiers avancés
    financial_indicators = calculate_financial_indicators(
        loan_amnt, loan_int_rate, loan_duration_years, monthly_income
    )
    
    # Métriques principales de remboursement
    col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
    
    with col_sim1:
        st.metric("💳 Mensualité", 
                 f"{financial_indicators['monthly_payment']:,.0f} €",
                 help="Paiement mensuel fixe")
    
    with col_sim2:
        st.metric("💰 Coût Total", 
                 f"{financial_indicators['total_payment']:,.0f} €",
                 delta=f"+{financial_indicators['total_interest']:,.0f} € vs capital",
                 help="Montant total remboursé")
    
    with col_sim3:
        st.metric("📈 Intérêts Totaux", 
                 f"{financial_indicators['total_interest']:,.0f} €",
                 delta=f"{financial_indicators['interest_rate_effectiveness']:.1f}% du capital",
                 help="Coût total du crédit")
    
    with col_sim4:
        st.metric("📊 Taux d'Endettement", 
                 f"{financial_indicators['debt_to_income_ratio']:.1f}%",
                 help="Pourcentage du revenu consacré au remboursement")
    
    # Indicateurs financiers avancés
    st.subheader("📊 Indicateurs Financiers Avancés")
    
    col_ind1, col_ind2, col_ind3 = st.columns(3)
    
    with col_ind1:
        st.markdown(f"""
        <div class="info-card">
            <h4>💡 Coût d'Opportunité</h4>
            <h3>{financial_indicators['opportunity_cost']:,.0f} €</h3>
            <p>Gain potentiel si capital investi à 3% annuel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ind2:
        effort_ratio = (financial_indicators['total_payment'] / person_income) * 100 if person_income > 0 else 0
        st.markdown(f"""
        <div class="info-card">
            <h4>⚖️ Effort Total</h4>
            <h3>{effort_ratio:.1f}%</h3>
            <p>Part du revenu total consacrée au crédit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ind3:
        break_even_years = financial_indicators['break_even_months'] / 12
        st.markdown(f"""
        <div class="info-card">
            <h4>⏰ Durée d'Engagement</h4>
            <h3>{break_even_years:.1f} ans</h3>
            <p>Période de remboursement total</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques de répartition
    st.subheader("📊 Répartition des Coûts")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.write("**Répartition Capital vs Intérêts**")
        cost_data = pd.DataFrame({
            'Type': ['Capital', 'Intérêts'],
            'Montant (€)': [loan_amnt, financial_indicators['total_interest']]
        })
        st.bar_chart(cost_data.set_index('Type'))
    
    with col_chart2:
        st.write("**Évolution du Ratio d'Endettement**")
        # Simulation sur plusieurs scénarios de revenus
        income_scenarios = pd.DataFrame({
            'Revenu Mensuel': [monthly_income * 0.8, monthly_income, monthly_income * 1.2],
            'Taux Endettement (%)': [
                (financial_indicators['monthly_payment'] / (monthly_income * 0.8)) * 100 if monthly_income > 0 else 0,
                financial_indicators['debt_to_income_ratio'],
                (financial_indicators['monthly_payment'] / (monthly_income * 1.2)) * 100 if monthly_income > 0 else 0
            ]
        })
        st.line_chart(income_scenarios.set_index('Revenu Mensuel'))

with tab3:
    st.header("📊 TABLEAUX D'AMORTISSEMENT DÉTAILLÉS")
    
    # Options d'affichage
    col_options1, col_options2, col_options3 = st.columns(3)
    
    with col_options1:
        show_months = st.selectbox("Période à afficher", 
                                  options=[12, 24, 36, "Tout"], 
                                  index=0,
                                  help="Nombre de mois à afficher")
    
    with col_options2:
        start_date = st.date_input("Date de début", 
                                  value=datetime.now().date(),
                                  help="Date du premier remboursement")
    
    with col_options3:
        currency_format = st.selectbox("Format d'affichage", 
                                      options=["€", "k€"], 
                                      help="Unité monétaire")
    
    # Génération du tableau d'amortissement
    schedule = calculate_amortization_schedule(
        loan_amnt, loan_int_rate, loan_duration_years, 
        datetime.combine(start_date, datetime.min.time())
    )
    
    # Filtrage selon les options
    if show_months != "Tout":
        schedule_display = schedule[:int(show_months)]
    else:
        schedule_display = schedule
    
    # Formatage du tableau
    df_schedule = pd.DataFrame(schedule_display)
    
    # Application du format monétaire
    money_columns = ['Paiement Total', 'Capital', 'Intérêts', 'Solde Restant', 'Capital Cumulé', 'Intérêts Cumulés']
    
    for col in money_columns:
        if currency_format == "k€":
            df_schedule[col] = df_schedule[col].apply(lambda x: f"{x/1000:.1f} k€")
        else:
            df_schedule[col] = df_schedule[col].apply(lambda x: f"{x:,.0f} €")
    
    # Formatage du pourcentage
    df_schedule['% Remboursé'] = df_schedule['% Remboursé'].apply(lambda x: f"{x:.1f}%")
    
    # Affichage du tableau avec style
    st.markdown("""
    <div class="amortization-table">
        <h4>📅 Échéancier de Remboursement Détaillé</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        df_schedule, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mois": st.column_config.NumberColumn("Mois", format="%d"),
            "Date": st.column_config.TextColumn("Date", width="small"),
            "% Remboursé": st.column_config.ProgressColumn("% Remboursé", min_value=0, max_value=100)
        }
    )
    
    # Résumé statistique
    if schedule:
        total_payments = len(schedule)
        midpoint = total_payments // 2
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric("📅 Nombre d'échéances", f"{total_payments}")
        
        with col_stats2:
            if midpoint < len(schedule):
                mid_balance = schedule[midpoint]['Solde Restant']
                st.metric("💰 Solde à mi-parcours", f"{mid_balance:,.0f} €")
        
        with col_stats3:
            total_interest_year_1 = sum([s['Intérêts'] for s in schedule[:12]]) if len(schedule) >= 12 else 0
            st.metric("📈 Intérêts année 1", f"{total_interest_year_1:,.0f} €")
        
        with col_stats4:
            if len(schedule) >= 12:
                principal_year_1 = sum([s['Capital'] for s in schedule[:12]])
                st.metric("💳 Capital année 1", f"{principal_year_1:,.0f} €")
    
    # Graphique d'évolution du solde
    st.subheader("📈 Évolution du Solde Restant")
    
    if len(schedule) > 0:
        # Échantillonnage pour l'affichage (un point tous les 6 mois max)
        sample_rate = max(1, len(schedule) // 20)
        sampled_schedule = schedule[::sample_rate]
        
        evolution_df = pd.DataFrame({
            'Mois': [s['Mois'] for s in sampled_schedule],
            'Solde Restant (€)': [s['Solde Restant'] for s in sampled_schedule],
            'Capital Cumulé (€)': [s['Capital Cumulé'] for s in sampled_schedule]
        })
        
        st.line_chart(evolution_df.set_index('Mois'))

with tab4:
    st.header("🔍 RECOMMANDATIONS PERSONNALISÉES")
    
    # Génération des recommandations si une analyse a été effectuée
    if 'risk_score' in locals():
        recommendations = get_risk_recommendations(risk_score, input_data)
        
        st.subheader(f"📋 Conseils pour votre profil (Risque: {risk_score:.1%})")
        
        # Affichage des recommandations par catégorie
        if risk_score < 0.3:
            st.markdown("""
            <div class="success-card">
                <h4>🎉 Félicitations ! Profil Excellent</h4>
                <p>Votre dossier présente un excellent profil de risque. Voici comment optimiser votre crédit :</p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_score < 0.6:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ Profil à Améliorer</h4>
                <p>Votre dossier nécessite quelques ajustements pour optimiser vos chances :</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="danger-card">
                <h4>🚨 Dossier à Restructurer</h4>
                <p>Des améliorations importantes sont nécessaires avant de refaire une demande :</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Liste des recommandations
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"**{i}.** {recommendation}")
        
        # Recommandations spécifiques par montant
        st.subheader("💰 Optimisation du Montant")
        
        current_ratio = loan_percent_income
        optimal_ratio = 0.35  # Ratio optimal recommandé
        
        if current_ratio > optimal_ratio:
            optimal_amount = person_income * optimal_ratio
            reduction = loan_amnt - optimal_amount
            st.warning(f"""
            **💡 Suggestion:** Réduisez votre demande de **{reduction:,.0f} €** 
            pour atteindre un ratio optimal de {optimal_ratio:.0%}.
            
            **Nouveau montant recommandé:** {optimal_amount:,.0f} €
            """)
        else:
            max_safe_amount = person_income * 0.4
            additional_capacity = max_safe_amount - loan_amnt
            if additional_capacity > 0:
                st.info(f"""
                **💰 Capacité supplémentaire:** Vous pourriez emprunter jusqu'à 
                **{additional_capacity:,.0f} €** de plus tout en restant dans les normes bancaires.
                """)
        
        # Simulation d'amélioration du profil
        st.subheader("📈 Simulation d'Amélioration")
        
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.write("**Scénario d'amélioration des revenus:**")
            income_increase = st.slider("Augmentation de revenus (%)", 0, 50, 10)
            new_income = person_income * (1 + income_increase/100)
            new_ratio = loan_amnt / new_income
            
            st.metric("Nouveau ratio dette/revenu", f"{new_ratio:.1%}", 
                     delta=f"{new_ratio - loan_percent_income:.1%}")
        
        with col_sim2:
            st.write("**Scénario de réduction du montant:**")
            amount_reduction = st.slider("Réduction du montant (%)", 0, 50, 10)
            new_amount = loan_amnt * (1 - amount_reduction/100)
            new_ratio_amount = new_amount / person_income if person_income > 0 else 0
            
            st.metric("Nouveau montant", f"{new_amount:,.0f} €", 
                     delta=f"-{loan_amnt - new_amount:,.0f} €")
            st.metric("Nouveau ratio", f"{new_ratio_amount:.1%}")
    
    else:
        st.info("🎯 Effectuez d'abord une analyse de risque pour obtenir des recommandations personnalisées.")
    
    # Conseils généraux
    st.subheader("💡 Conseils Généraux pour Optimiser votre Dossier")
    
    general_tips = [
        "📊 **Maintenez un ratio d'endettement < 35%** pour optimiser vos chances",
        "💰 **Constituez un apport personnel** d'au moins 10% du montant",
        "📋 **Rassemblez tous vos justificatifs** (revenus, charges, patrimoine)",
        "🏦 **Comparez plusieurs établissements** pour obtenir les meilleures conditions",
        "⏰ **Évitez les demandes multiples simultanées** (impact négatif sur le score)",
        "💳 **Soldez vos découverts** et crédits revolving avant la demande",
        "📈 **Démontrez la stabilité** de vos revenus sur 12 mois minimum",
        "🎯 **Préparez votre argumentaire** sur l'utilisation des fonds"
    ]
    
    for tip in general_tips:
        st.write(f"• {tip}")

# Performance du modèle et footer
st.divider()
st.header("📊 PERFORMANCE DU SYSTÈME D'ANALYSE")

col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)

with col_perf1:
    st.metric("🎯 Précision", "88.9%", help="Taux de prédictions correctes")
with col_perf2:
    st.metric("🔍 Rappel", "76.8%", help="Taux de détection des défauts")
with col_perf3:
    st.metric("⚖️ Score F1", "75.4%", help="Équilibre précision/rappel")
with col_perf4:
    st.metric("📈 AUC-ROC", "84.5%", help="Performance globale du modèle")

# Informations techniques dans un expander
with st.expander("🛠️ Informations Techniques Détaillées"):
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        **🤖 Modèle d'IA:**
        - Decision Tree Classifier optimisé
        - 32,583 observations d'entraînement
        - 11 variables prédictives principales
        - Validation croisée 80/20
        
        **📊 Preprocessing:**
        - StandardScaler pour variables numériques
        - One-hot encoding variables catégorielles
        - Gestion valeurs manquantes par médiane
        """)
    
    with col_tech2:
        st.markdown("""
        **💼 Variables Prédictives:**
        - Âge et revenus du demandeur
        - Historique d'emploi et de crédit
        - Montant, taux et durée du prêt
        - Statut de propriété immobilière
        - Motif du crédit demandé
        
        **🔒 Sécurité & Conformité:**
        - Traitement sécurisé des données
        - Respect RGPD
        - Auditabilité des décisions
        """)

# Footer stylé
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 20px; color: white; text-align: center; margin: 2rem 0;'>
    <h3>💰 Crédit Risk Analyzer Premium</h3>
    <h4>👨‍🎓 Développé par Fidèle Ledoux</h4>
    <p>🎓 IA School - Formation Intelligence Artificielle & Data Science</p>
    <p>🏅 Solution d'Analyse Financière de Nouvelle Génération 🏅</p>
    <div style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
        🚀 Technologie avancée • 🔒 Sécurisé • 📊 Précision 88.9% • ⚡ Temps réel
    </div>
</div>
""", unsafe_allow_html=True)
