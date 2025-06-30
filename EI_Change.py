# Load libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from ipywidgets import Text, Output, VBox, Button, Layout
from IPython.display import display

# Read the Excel file
df = pd.read_excel('Eco_Intel2_clean2.xlsx')
# Remove rows with missing values in key columns
model_data = df[['argument', 'change']]
# Remove any rows where 'argument' is missing or the index is not numeric
model_data = model_data[model_data['argument'].notna()]

def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

model_data['argument_clean'] = model_data['argument'].apply(preprocess_text)

# Create TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words=None,  # You may want to add Spanish stop words
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(model_data['argument_clean'])
y = model_data['change']

# Train models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    results[name] = {
        'model': model,
        'mse': mse,
        'r2': r2,
        'predictions': y_pred
    }

# Feature importance for Random Forest
rf_model = results['Random Forest']['model']
feature_names = vectorizer.get_feature_names_out()
importances = rf_model.feature_importances_

# Get top 20 most important features
top_indices = np.argsort(importances)[-20:]
top_features = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

# Function to predict change score for new text
def predict_change_score(text, model_name='Random Forest'):
    """Predict change score for a given argument text"""
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = results[model_name]['model'].predict(text_vector)[0]
    return round(prediction, 2)

# Interactive prediction function
def analyze_argument(argument_text):
    """Analyze an argument and predict change score"""
    score = predict_change_score(argument_text)
    
    if score >= 8:
        level = "Alto impacto"
        color = "🟢"
    elif score >= 6:
        level = "Impacto moderado"
        color = "🟡"
    elif score >= 4:
        level = "Impacto bajo"
        color = "🟠"
    else:
        level = "Muy bajo impacto"
        color = "🔴"
    
    print(f"\n{color} Análisis del argumento:")
    print(f"Texto: '{argument_text[:100]}...' " if len(argument_text) > 100 else f"Texto: '{argument_text}'")
    print(f"Puntuación de cambio predicha: {score}/10")
    print(f"Nivel de impacto: {level}")
    
    return score

# Interfaz interactiva
entrada = Text(
    description='Question 14:',
    placeholder='Da una razón de tu cambio de mentalidad y sostenibilidad.',
    layout=Layout(width='600px')
)
out = Output()
boton = Button(
    description='Analizar cambio Inteligencia Ecológica',
    layout=Layout(width='300px')  # Set button width here
)

def analizar_ie(b):
    with out:
        out.clear_output()
        analyze_argument(entrada.value)

boton.on_click(analizar_ie)
display(VBox([entrada, boton, out]))