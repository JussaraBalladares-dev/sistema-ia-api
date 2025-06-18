from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from typing import List

# Inicializa FastAPI
app = FastAPI()

# Simula el LabelEncoder de habilidades
habilidad_encoder = {"frontend": 0, "backend": 1, "QA": 2}  # Asegúrate de que coincida con el entrenamiento

# Request model
class TareaEntrenamiento(BaseModel):
    valor_habilidad: int  # categórico: 1=backend, 2=frontend, etc
    valor_prioridad : float  # más alto es mejor
    valor_tarea_clara: float  # más alto es mejor
    valor_carga_trabajo: float  # más alto es mejor
    valor_adecuado_habilidades: float  # más alto es mejor
    tiempo_estimado: float
    # tiempo_real: float
    # porcentaje_avance: float  # 0 a 100
    # id_estado_tarea: int  # 1=pendiente, 2=en curso, 3=terminado
    id_usuario_asignado: int  # target

class Tarea(BaseModel):
    valor_habilidad: int
    valor_prioridad: float
    # valor_tarea_clara: float
    # valor_carga_trabajo: float
    # valor_adecuado_habilidades: float
    tiempo_estimado: float
    # tiempo_real: float
    # porcentaje_avance: float
    # id_estado_tarea: int

# Cargar modelo entrenado
model = None  # Inicializamos en None

@app.post("/entrenar-modelo/")
def entrenar_modelo(listaTareas: List[TareaEntrenamiento]):
    global model  # Indicamos que queremos modificar la variable global

    data = pd.DataFrame([t.dict() for t in listaTareas])
    
    # One-Hot Encoding para 'valor_habilidad'
    data = pd.get_dummies(data, columns=['valor_habilidad'], prefix='habilidad')

    # Calcular feedback promedio como peso
    data["peso_feedback"] = (
        data["valor_tarea_clara"].fillna(3) +
        data["valor_carga_trabajo"].fillna(3) +
        data["valor_adecuado_habilidades"].fillna(3)
    ) / 3

    # Definimos features relevantes
    feature_cols = [
        'valor_prioridad',
        # 'valor_tarea_clara',
        # 'valor_carga_trabajo',
        # 'valor_adecuado_habilidades',
        'tiempo_estimado',
        # 'tiempo_real',
        # 'porcentaje_avance',
        # 'id_estado_tarea',
    ] + [col for col in data.columns if col.startswith('habilidad_')]

    X = data[feature_cols]
    y = data['id_usuario_asignado']
    weights = data["peso_feedback"]
    
    print(data)
    data.to_csv("data_entrenamiento.csv", index=False)

    # Entrenamos modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y, sample_weight=weights)

    joblib.dump(model, "modelo_asignacion.pkl")

    # Cargar modelo entrenado
    model = joblib.load("modelo_asignacion.pkl")

    return {"success": True, "message": "Modelo entrenado correctamente", "features_usadas": feature_cols}

@app.post("/asignar-tarea/")
def asignar_tarea(tarea: Tarea):
    global model

    if model is None:
        model = joblib.load("modelo_asignacion.pkl")

    # One-hot encoding manual según lo entrenado
    habilidades = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Asegúrate de usar los mismos valores del entrenamiento
    habilidad_vector = {f"habilidad_{h}": 0 for h in habilidades}

    if tarea.valor_habilidad not in habilidades:
        return {"error": "Habilidad no reconocida"}

    habilidad_vector[f"habilidad_{tarea.valor_habilidad}"] = 1

    # Crear el DataFrame con los valores requeridos
    entrada = pd.DataFrame([{
        "valor_prioridad": tarea.valor_prioridad,
        # "valor_tarea_clara": tarea.valor_tarea_clara,
        # "valor_carga_trabajo": tarea.valor_carga_trabajo,
        # "valor_adecuado_habilidades": tarea.valor_adecuado_habilidades,
        "tiempo_estimado": tarea.tiempo_estimado,
        **habilidad_vector
    }])
    
    print(tarea)
    print(entrada)

    # Asegurar que columnas faltantes se agreguen con cero (en caso de que haya menos habilidades)
    for col in model.feature_names_in_:
        if col not in entrada.columns:
            entrada[col] = 0

    entrada = entrada[model.feature_names_in_]  # Reordenar las columnas

    prediccion = model.predict(entrada)[0]

    return {"usuario_sugerido": int(prediccion)}
