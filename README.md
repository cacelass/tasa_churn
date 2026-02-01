# Churn Prediction

Proyecto para **predecir la probabilidad de abandono de clientes** (tasa de churn) usando técnicas de **Machine Learning supervisado**. El objetivo es identificar clientes con riesgo de abandonar y tomar acciones preventivas para mejorar la retención.

---

## Descripción del problema

En muchos negocios, el **abandono de clientes** tiene un impacto directo en los ingresos y en la estabilidad del negocio. La tasa de churn mide el porcentaje de clientes que dejan de usar el servicio durante un periodo determinado.

El desafío consiste en predecir, a partir de datos históricos y de comportamiento del cliente:

- Si un cliente es probable que abandone (churn = 1) o permanezca (churn = 0).  
- La **probabilidad de abandono** para cada cliente, no solo una predicción binaria.  

Contamos con información histórica sobre clientes, incluyendo:

- Datos demográficos: edad, género, segmento, antigüedad.  
- Datos de uso: frecuencia de compra/uso, tickets promedio, productos contratados.  
- Interacciones con el servicio: quejas, llamadas a soporte, pagos atrasados.

---

## Solución propuesta

El enfoque principal es **aprendizaje supervisado**, donde entrenamos modelos con datos etiquetados (clientes que han abandonado o permanecido) para que aprendan patrones predictivos.  

### Modelos considerados

Para evaluar y seleccionar el mejor modelo, se probarán distintos algoritmos supervisados:

- **Logistic Regression:** proporciona probabilidades explícitas y sirve como baseline.  
- **Decision Tree:** captura relaciones no lineales entre variables y es interpretable.  
- **Random Forest:** ensemble robusto que reduce overfitting y mejora precisión.  
- **K-Nearest Neighbors (KNN):** modelo simple basado en similitud entre clientes.  
- **Support Vector Machine (SVM):** bueno para separar clases cuando los datos no son lineales.  
- **Gradient Boosting / LightGBM:** suele ser el más preciso en problemas de churn.

### Métricas de evaluación

- **Clasificación binaria:** Accuracy, Precision, Recall, F1-score.  
- **Probabilidades de churn:** ROC-AUC, Log Loss o Brier Score para evaluar la calidad de las probabilidades.  

---

## Pipeline del proyecto

1. **Exploración de datos:** análisis de distribuciones, valores faltantes y correlaciones (`pandas`, `seaborn`, `missingno`).  
2. **Preprocesamiento:** limpieza, codificación de variables categóricas, escalado de features, imputación de faltantes.  
3. **División train/test** y validación cruzada para asegurar robustez.  
4. **Entrenamiento de modelos supervisados** y comparación de desempeño.  
5. **Selección del modelo final** según métricas y calibración de probabilidades.  
6. **Interpretación de resultados:** importancia de features, identificación de clientes con alto riesgo de churn.  

---

## Tecnologías y librerías

- **Python ≥ 3.11**  
- **Análisis y visualización de datos:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`  
- **Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`, `keras`, `tensorflow`  
- **Exploración de datos faltantes y limpieza:** `pyjanitor`, `missingno`  

---

## Objetivo

Construir un **framework reproducible y escalable** para predecir la probabilidad de churn de clientes, proporcionando información accionable para mejorar la retención y optimizar estrategias de negocio.

---
## Project Organization

        ├── LICENSE
        ├── tasks.py           <- Archivo con tareas que puedes ejecutar con comandos como `notebook`.
        ├── README.md          <- Guía principal para desarrolladores que trabajen con este proyecto.
        ├── install.md         <- Instrucciones paso a paso para instalar y configurar el entorno.
        ├── data
        │   ├── external       <- Datos obtenidos de fuentes externas.
        │   ├── interim        <- Datos intermedios, ya transformados pero no finales.
        │   ├── processed      <- Datos finales listos para ser usados en modelos.
        │   └── raw            <- Datos originales sin modificar.
        │
        ├── models             <- Modelos entrenados, guardados y sus predicciones o reportes.
        │
        ├── notebooks          <- Notebooks de Jupyter. Se nombran con un número (para ordenar),
        │                         iniciales del autor y una breve descripción, por ejemplo:
        │                         `1.0-jqp-exploracion-inicial`.
        │
        ├── references         <- Documentación de apoyo: diccionarios de datos, manuales, etc.
        │
        ├── reports            <- Resultados generados en formatos como HTML, PDF o LaTeX.
        │   └── figures        <- Gráficos e imágenes usados en los reportes.
        │
        ├── pyproject.toml     <- Archivo con las dependencias necesarias para reproducir el entorno.
        │
        ├── .here              <- Archivo que indica el directorio raíz del proyecto.
        │
        └── tasa_churn               <- Código fuente principal del proyecto.
            ├── __init__.py    <- Indica que este directorio es un módulo de Python.
            │
            ├── data           <- Scripts para descargar, generar o preparar datos.
            │   └── make_dataset.py
            │
            ├── features       <- Scripts para convertir datos crudos en características útiles.
            │   └── build_features.py
            │
            ├── models         <- Scripts para entrenar modelos y generar predicciones.
            │   ├── predict_model.py
            │   └── train_model.py
            │
            ├── utils          <- Funciones auxiliares para tareas comunes del proyecto.
            │   └── paths.py   <- Funciones para manejar rutas de archivos dentro del proyecto.
            │
            └── visualization  <- Scripts para crear visualizaciones y gráficos de resultados.
                └── visualize.py

---

## Autor

**Alejandro Cancelas Chapela**