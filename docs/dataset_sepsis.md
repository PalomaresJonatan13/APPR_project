# Dataset de sepsis de PhysioNet 2019

Este documento resume el contexto, la estructura de los datos y la evaluación del desafío PhysioNet/Computing in Cardiology 2019 para un proyecto de predicción temprana.

- **Página oficial:** https://physionet.org/content/challenge-2019/1.0.0/
- **Descripción del desafío:** https://moody-challenge.physionet.org/2019/
- **DOI del dataset:** https://doi.org/10.13026/v64v-d857

## Resumen

El desafío busca anticipar la sepsis a partir de mediciones de pacientes en cuidados intensivos. Solicita estimaciones de riesgo y alertas binarias cada hora, con el objetivo de advertir seis horas antes del inicio definido y limitar las falsas alarmas. [1]

## Contexto

La sepsis es una respuesta perjudicial a una infección que puede producir disfunción orgánica. El desafío aplica criterios de Sepsis-3 mediante sospecha de infección y un aumento mínimo de dos puntos en SOFA. La oportunidad de las alertas importa, pero las innecesarias también consumen recursos clínicos. [1]

## Descripción de los datos

Los datos públicos de entrenamiento incluyen 40 336 pacientes: 20 336 en A y 20 000 en B, procedentes de dos sistemas hospitalarios. La competencia también utilizó datos de prueba ocultos. Cada paciente tiene un archivo PSV delimitado por `|`, con filas horarias, 40 predictores y `SepsisLabel`. [1]

Los predictores comprenden ocho signos vitales, 26 variables de laboratorio y seis variables demográficas o administrativas. `ICULOS` expresa las horas desde el ingreso a cuidados intensivos. `NaN` indica que no existe una medición registrada de esa variable en esa hora. [1]

En pacientes que desarrollan sepsis, `SepsisLabel` pasa a 1 seis horas antes del inicio definido y permanece en 1 después. En los demás, todas las etiquetas son 0. Este desplazamiento establece una tarea de alerta temprana, no un diagnóstico anterior. [1]

## Referencias

[1] Reyna, M., et al. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, versión 1.0.0. PhysioNet. https://doi.org/10.13026/v64v-d857

[2] PhysioNet. *The PhysioNet/Computing in Cardiology Challenge 2019*. Descripción del desafío y de su evaluación. https://moody-challenge.physionet.org/2019/

Fecha de consulta: 15 de octubre de 2026. El contenido es un resumen de las fuentes, no una reproducción literal.
