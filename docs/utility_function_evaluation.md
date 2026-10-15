# Evaluación y función de utilidad

La utilidad oficial considera el momento de la alerta. En pacientes con sepsis, recompensa alertas entre 12 horas antes y tres horas después del inicio, con recompensa máxima seis horas antes. Penaliza alertas excesivamente anticipadas y avisos tardíos u omitidos. En pacientes sin sepsis, penaliza falsas alarmas; no alertar aporta utilidad cero. [2]

Las utilidades se suman por paciente y hora, y se normalizan respecto de un predictor que nunca alerta y otro óptimo. El valor 0 corresponde a no emitir alertas y 1 a predicciones óptimas; pueden obtenerse valores negativos. Un resultado mayor es mejor. Existe código oficial de evaluación en Python. [2]

En cada hora solo pueden utilizarse mediciones actuales y pasadas. La competencia evaluó algoritmos con registros ocultos; los resultados de una partición propia no son directamente comparables con su clasificación oficial. [2]