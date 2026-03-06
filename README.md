# 🤖 Predicción de Evasión de Clientes (Churn) - Telecom X | Parte 2: Machine Learning

## 🎯 Propósito del Proyecto
Este proyecto representa la Fase 2 del Desafío de Data Science de Alura LATAM para la empresa Telecom X. Tras completar el pipeline de ETL y Análisis Exploratorio (EDA) en la Fase 1, el objetivo actual es construir, evaluar y comparar modelos de Machine Learning capaces de **predecir qué clientes tienen mayor probabilidad de cancelar sus servicios (Churn)**, extrayendo insights accionables para la retención.

---

## ⚙️ Pipeline de Machine Learning
El desarrollo siguió un riguroso proceso de Ciencia de Datos para garantizar modelos robustos y evitar sesgos:

1. **Preparación y Limpieza (Data Munging):** - Se eliminaron variables irrelevantes (`customerid`).
   - Se aplicó **One-Hot Encoding** (`pd.get_dummies`) para transformar variables categóricas al formato numérico exigido por los algoritmos.
2. **Balanceo de Clases:**
   - Se detectó un desequilibrio en la variable objetivo (aprox. 73% retención vs 26% cancelación).
   - Se implementó la técnica **SMOTE** (Synthetic Minority Over-sampling Technique) exclusivamente en el conjunto de entrenamiento para generar ejemplos sintéticos de la clase minoritaria y evitar el sobreajuste hacia la clase mayoritaria.
3. **Estandarización:**
   - Se utilizó `StandardScaler` para normalizar las características antes de alimentar modelos basados en distancias (como la Regresión Logística).

---

## 🧠 Modelos Implementados y Evaluación
Se entrenaron y compararon dos enfoques distintos:

* **Regresión Logística (Modelo Paramétrico):** Se alimentó con datos estandarizados. Resultó ser el modelo más robusto para este caso de negocio, logrando un excelente equilibrio, destacando en su capacidad para identificar correctamente a los clientes en riesgo (alto Recall).
* **Random Forest (Modelo de Ensamblaje):** Se alimentó con datos sin estandarizar. Aunque mostró métricas iniciales fuertes, el análisis crítico reveló una tendencia al *Overfitting* al memorizar los patrones de entrenamiento (especialmente los datos sintéticos de SMOTE), reduciendo su capacidad de generalización en el set de prueba.

---

## 📊 Insights y Conclusiones de Negocio
A través del análisis de la importancia de variables (coeficientes y reducción de impureza), los modelos revelaron la "fórmula de la evasión" en Telecom X:

1. **El Factor del Corto Plazo (`tenure`):** Es la variable más crítica. El riesgo de cancelación es altísimo en los primeros meses y disminuye significativamente a medida que el cliente madura.
2. **Contratos Flexibles (`contract_Mensual`):** La falta de ataduras contractuales facilita la salida rápida de los usuarios insatisfechos.
3. **Sensibilidad al Precio (`total` / `monthly`):** Altos cargos recurrentes generan fricción y motivan la deserción.

### 💡 Estrategias Propuestas:
* **Escudo de Primer Año (Onboarding):** Implementar soporte prioritario y seguimiento proactivo durante los primeros 6 meses de vida del cliente.
* **Campaña de "Anualización":** Ofrecer upgrades sin costo (ej. mayor velocidad o canales extra) a cambio de migrar de un contrato mensual a uno anual.
* **Auditoría Técnica:** Revisar la calidad del servicio de Fibra Óptica, ya que apareció recurrentemente como un factor asociado a la cancelación.

---

## 🚀 Cómo Ejecutar este Proyecto
**Requisitos Previos:**
Asegúrate de tener instaladas las siguientes librerías:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
