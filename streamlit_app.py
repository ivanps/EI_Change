# Streamlit app for Ecological Intelligence Change Prediction
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Análisis de Inteligencia Ecológica",
    page_icon="🌱",
    layout="wide"
)

# Cache the model training to avoid retraining on every interaction
@st.cache_data
def load_and_train_models():
    # Read the Excel file
    df = pd.read_excel('Eco_Intel2_clean2.xlsx')
    # Remove rows with missing values in key columns
    model_data = df[['argument', 'change']]
    # Remove any rows where 'argument' is missing
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

    return vectorizer, results, top_features, top_importances, model_data

def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def predict_change_score(text, vectorizer, results, model_name='Random Forest'):
    """Predict change score for a given argument text"""
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = results[model_name]['model'].predict(text_vector)[0]
    return round(prediction, 2)

def analyze_argument(argument_text, vectorizer, results):
    """Analyze an argument and predict change score"""
    score = predict_change_score(argument_text, vectorizer, results)
    
    if score >= 8:
        level = "Alto impacto"
        color = "🟢"
        color_code = "green"
    elif score >= 6:
        level = "Impacto moderado"
        color = "🟡"
        color_code = "orange"
    elif score >= 4:
        level = "Impacto bajo"
        color = "🟠"
        color_code = "orange"
    else:
        level = "Muy bajo impacto"
        color = "🔴"
        color_code = "red"
    
    return score, level, color, color_code

# Main app
def main():
    st.title("🌱 Análisis de Inteligencia Ecológica")
    st.markdown("### Predicción de Cambio de Mentalidad y Sostenibilidad")
    
    # Load and train models
    try:
        vectorizer, results, top_features, top_importances, model_data = load_and_train_models()
        st.success("✅ Modelos cargados y entrenados exitosamente")
        
        # Show model performance
        with st.expander("📊 Rendimiento de los Modelos"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Random Forest")
                st.metric("R² Score", f"{results['Random Forest']['r2']:.3f}")
                st.metric("MSE", f"{results['Random Forest']['mse']:.3f}")
                
            with col2:
                st.subheader("Linear Regression")
                st.metric("R² Score", f"{results['Linear Regression']['r2']:.3f}")
                st.metric("MSE", f"{results['Linear Regression']['mse']:.3f}")
        
        # Show feature importance
        with st.expander("🔍 Características Más Importantes"):
            st.subheader("Top 20 Palabras/Frases Más Influyentes")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(len(top_features)), top_importances)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Importancia')
            ax.set_title('Características Más Importantes del Modelo Random Forest')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Main prediction interface
        st.markdown("---")
        st.subheader("💭 Analiza tu Argumento")
        
        # Text input
        argument_text = st.text_area(
            "Question 14: Da una razón de tu cambio de mentalidad y sostenibilidad:",
            placeholder="Escribe aquí tu argumento sobre el cambio de mentalidad hacia la sostenibilidad...",
            height=150
        )
        
        # Model selection
        model_choice = st.selectbox(
            "Selecciona el modelo:",
            options=list(results.keys()),
            index=0
        )
        
        # Analyze button
        if st.button("🔍 Analizar Cambio de Inteligencia Ecológica", type="primary"):
            if argument_text.strip():
                score, level, emoji, color_code = analyze_argument(argument_text, vectorizer, results)
                
                # Display results
                st.markdown("### 📋 Resultados del Análisis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Puntuación de Cambio",
                        value=f"{score}/10"
                    )
                
                with col2:
                    st.markdown(f"**Nivel de Impacto:**")
                    st.markdown(f"{emoji} {level}")
                
                with col3:
                    # Progress bar
                    progress_value = score / 10
                    st.markdown("**Progreso:**")
                    st.progress(progress_value)
                
                # Detailed analysis
                st.markdown("### 📝 Análisis Detallado")
                
                if score >= 8:
                    st.success(f"🎉 ¡Excelente! Tu argumento muestra un muy alto potencial de cambio hacia la sostenibilidad.")
                elif score >= 6:
                    st.info(f"👍 Buen trabajo. Tu argumento demuestra un impacto moderado hacia el cambio sostenible.")
                elif score >= 4:
                    st.warning(f"⚠️ Tu argumento tiene potencial, pero podría fortalecerse para mayor impacto.")
                else:
                    st.error(f"💡 Considera expandir tu argumento con más elementos relacionados con la sostenibilidad.")
                
                # Show the argument
                st.markdown("**Tu argumento:**")
                st.write(f"'{argument_text}'")
                
            else:
                st.warning("⚠️ Por favor, ingresa un argumento para analizar.")
        
        # Statistics about the dataset
        with st.expander("📈 Estadísticas del Dataset"):
            st.subheader("Información del Dataset")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Argumentos", len(model_data))
            
            with col2:
                st.metric("Puntuación Media", f"{model_data['change'].mean():.2f}")
            
            with col3:
                st.metric("Desviación Estándar", f"{model_data['change'].std():.2f}")
            
            # Distribution plot
            st.subheader("Distribución de Puntuaciones de Cambio")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(model_data['change'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Puntuación de Cambio')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Puntuaciones de Cambio en el Dataset')
            plt.tight_layout()
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.error("❌ Error: No se pudo encontrar el archivo 'Eco_Intel2_clean2.xlsx'. Asegúrate de que el archivo esté en el directorio correcto.")
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")

if __name__ == "__main__":
    main()
