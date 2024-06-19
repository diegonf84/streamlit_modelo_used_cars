import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configurar la página
st.set_page_config(
    page_title="Predicción de Autos Usados",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto",
)

# Cargar el modelo
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Función para realizar predicciones
def make_predictions(model, data):
    return model.predict(data)

# Función para modificar los MPG
def get_mpg(x):
    x = str(x)
    if x == 'nan':
        return np.nan
    elif len(x) <= 2:
        return float(x)
    else:
        return (float(x.split('-')[0])+float(x.split('-')[1]))/2

# Título de la aplicación
st.title('Predicción de Modelo con Streamlit')

# Paso 1: Subir el archivo .csv
uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        try:
            # Leer el archivo .csv en un DataFrame de pandas
            df = pd.read_csv(uploaded_file)

            # Paso 2: Realizar modificaciones al DataFrame
            # Aquí puedes agregar las modificaciones que necesites hacer al DataFrame
            if 'mpg' in df.columns:
                if df['mpg'].dtype =='O':
                    df['mpg'] = df['mpg'].apply(get_mpg)

            if 'Unnamed: 0' in df.columns:
                df.drop(columns = 'Unnamed: 0', inplace=True)

            st.write("Datos cargados:")
            st.write(df)
            
            # Paso 3: Realizar la predicción al presionar el botón
            if st.button('Realizar Predicción'):
                model = load_model('modelo_autos_usados_v1.pkl')  # Cargar el modelo previamente guardado
                predictions = make_predictions(model, df)
                df['precio_estimado'] = predictions

                # Guardar las predicciones en un archivo .csv
                fecha = datetime.now().strftime("%Y-%m-%d")
                output_file = f'prediccion_{fecha}.csv'
                #df.to_csv(output_file, index=False)

                # Mostrar un enlace para descargar el archivo
                st.write("Predicciones realizadas. Para obtener el archivo debe clikear el siguiente link:")
                st.download_button(
                    label="Descargar predicciones",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=output_file,
                    mime='text/csv'
                )
        except Exception as e:
            st.error("No se pudo realizar la predicción por errores en el archivo")
    else:
        st.error("Por favor, sube un archivo en formato .csv")