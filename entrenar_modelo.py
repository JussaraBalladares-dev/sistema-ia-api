import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dataset simulado
data = pd.DataFrame([
    {"prioridad": 5, "habilidad_requerida": "frontend", "tiempo_estimado": 4, "carga_trabajo_actual": 10, "usuario_asignado": 1},
    {"prioridad": 3, "habilidad_requerida": "backend", "tiempo_estimado": 6, "carga_trabajo_actual": 8, "usuario_asignado": 2},
    {"prioridad": 1, "habilidad_requerida": "QA", "tiempo_estimado": 3, "carga_trabajo_actual": 5, "usuario_asignado": 3},
])

# Codificar habilidades
data['habilidad_requerida'] = data['habilidad_requerida'].map({"frontend": 0, "backend": 1, "QA": 2})

X = data[['prioridad', 'habilidad_requerida', 'tiempo_estimado', 'carga_trabajo_actual']]
y = data['usuario_asignado']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "modelo_asignacion.pkl")
