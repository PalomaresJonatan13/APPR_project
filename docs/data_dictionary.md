# Diccionario de variables de PhysioNet 2019

Las columnas siguientes corresponden a los archivos PSV originales. La columna de unidades reproduce la documentación oficial; las observaciones al final señalan unidades o escalas que requieren verificación en los datos.

## Signos vitales

| Columna | Unidad publicada | Descripción |
|---|---|---|
| `HR` | latidos/min | **Frecuencia cardíaca:** número de latidos del corazón por minuto. |
| `O2Sat` | % | **Saturación periférica de oxígeno:** estimación del porcentaje de hemoglobina que transporta oxígeno mediante un pulsioxímetro. |
| `Temp` | °C | **Temperatura corporal.** |
| `SBP` | mmHg | **Presión arterial sistólica:** presión máxima durante la contracción del corazón. |
| `MAP` | mmHg | **Presión arterial media:** presión promedio durante un ciclo cardíaco; ayuda a valorar el aporte de sangre a los órganos. |
| `DBP` | mmHg | **Presión arterial diastólica:** presión durante la relajación del corazón entre latidos. |
| `Resp` | respiraciones/min | **Frecuencia respiratoria:** número de respiraciones por minuto. |
| `EtCO2` | mmHg | **CO₂ al final de la espiración:** presión parcial del dióxido de carbono al terminar de exhalar; informa sobre la ventilación. |

## Variables de laboratorio

| Columna | Unidad publicada | Descripción |
|---|---|---|
| `BaseExcess` | mmol/L | **Exceso de bases:** estima el componente metabólico del equilibrio ácido-base. Un valor negativo indica déficit de bases; uno positivo, exceso. |
| `HCO3` | mmol/L | **Bicarbonato:** sustancia que contribuye a regular la acidez de la sangre. |
| `FiO2` | %; verificar escala | **Fracción inspirada de oxígeno:** proporción de oxígeno en el gas que respira el paciente. Puede aumentar al administrar oxígeno suplementario. Véase la nota sobre escalas. |
| `pH` | Sin unidad | **Acidez o alcalinidad sanguínea:** valores menores indican mayor acidez y mayores, mayor alcalinidad. Es una magnitud logarítmica. |
| `PaCO2` | mmHg | **Presión parcial arterial de CO₂:** refleja el dióxido de carbono disuelto en sangre arterial y aporta información sobre la ventilación. |
| `SaO2` | % | **Saturación arterial de oxígeno:** porcentaje de hemoglobina que transporta oxígeno en sangre arterial. Se diferencia de `O2Sat` por el origen y método de medición. |
| `AST` | IU/L | **Aspartato aminotransferasa:** enzima del hígado, músculos y otros tejidos; su elevación puede reflejar lesión celular. |
| `BUN` | mg/dL | **Nitrógeno ureico en sangre:** nitrógeno contenido en la urea, un desecho del metabolismo de proteínas. Aporta información sobre función renal e hidratación. |
| `Alkalinephos` | IU/L | **Fosfatasa alcalina:** enzima relacionada principalmente con hígado, vías biliares y huesos. |
| `Calcium` | mg/dL | **Calcio sanguíneo:** mineral necesario para la contracción muscular, la transmisión nerviosa y la coagulación. |
| `Chloride` | mmol/L | **Cloruro sanguíneo:** electrolito que participa en el equilibrio de líquidos y ácido-base. |
| `Creatinine` | mg/dL | **Creatinina:** desecho del metabolismo muscular eliminado principalmente por los riñones; se utiliza para evaluar la función renal. |
| `Bilirubin_direct` | mg/dL | **Bilirrubina directa o conjugada:** pigmento procesado por el hígado para su eliminación en la bilis. |
| `Glucose` | mg/dL | **Glucosa sanguínea:** concentración de azúcar en sangre, fuente de energía para las células. |
| `Lactate` | mg/dL; verificar | **Lactato sanguíneo:** producto del metabolismo que puede elevarse por alteraciones en el aporte de oxígeno, estrés metabólico o menor eliminación, entre otras causas. |
| `Magnesium` | mmol/dL; verificar | **Magnesio sanguíneo:** mineral que interviene en reacciones celulares y en la función muscular y nerviosa. |
| `Phosphate` | mg/dL | **Fosfato sanguíneo:** participa en la producción de energía celular y en la estructura de huesos y membranas. |
| `Potassium` | mmol/L | **Potasio sanguíneo:** electrolito esencial para la actividad eléctrica del corazón y la función muscular y nerviosa. |
| `Bilirubin_total` | mg/dL | **Bilirrubina total:** suma de bilirrubina directa e indirecta; informa sobre su procesamiento hepático y la degradación de glóbulos rojos. |
| `TroponinI` | ng/mL | **Troponina I cardíaca:** proteína utilizada como marcador de lesión del músculo cardíaco. Su elevación no implica por sí sola un infarto. |
| `Hct` | % | **Hematocrito:** porcentaje del volumen sanguíneo ocupado por glóbulos rojos. |
| `Hgb` | g/dL | **Hemoglobina:** concentración de la proteína de los glóbulos rojos que transporta oxígeno. |
| `PTT` | s | **Tiempo de tromboplastina parcial:** tiempo de coagulación bajo condiciones de laboratorio; evalúa parte del sistema de coagulación. |
| `WBC` | 10³ células/µL | **Recuento de leucocitos:** cantidad de glóbulos blancos, que participan en la respuesta inmunitaria. Un valor de 8 representa aproximadamente 8000 células/µL. |
| `Fibrinogen` | mg/dL | **Fibrinógeno:** proteína que se transforma en fibrina durante la formación de coágulos. |
| `Platelets` | 10³ células/µL | **Recuento de plaquetas:** cantidad de fragmentos celulares que participan en la coagulación y el control del sangrado. Un valor de 200 representa aproximadamente 200 000 plaquetas/µL. |

## Variables demográficas y administrativas

| Columna | Unidad o codificación | Descripción |
|---|---|---|
| `Age` | años | **Edad:** el valor `100` representa a pacientes de 90 años o más como medida de desidentificación. |
| `Gender` | Categoría binaria: 0 o 1 | **Sexo según el dataset:** `0` corresponde a femenino y `1` a masculino. |
| `Unit1` | Indicador binario: 0 o 1 | **Indicador de UCI médica (`MICU`):** unidad que atiende principalmente problemas médicos que requieren cuidados intensivos. `1` indica pertenencia; un valor faltante no debe interpretarse automáticamente como `0`. |
| `Unit2` | Indicador binario: 0 o 1 | **Indicador de UCI quirúrgica (`SICU`):** unidad para pacientes que requieren cuidados intensivos relacionados con cirugía. `1` indica pertenencia; un valor faltante no debe interpretarse automáticamente como `0`. |
| `HospAdmTime` | h | **Momento del ingreso hospitalario respecto al ingreso en UCI:** normalmente negativo. `-48` significa que el paciente ingresó al hospital 48 horas antes de entrar en UCI. |
| `ICULOS` | h | **Horas transcurridas desde el ingreso en UCI:** ubica temporalmente cada registro. No expresa la duración total futura de la estancia. |

## Variable objetivo

| Columna | Unidad o codificación | Descripción |
|---|---|---|
| `SepsisLabel` | Etiqueta binaria: 0 o 1 | En pacientes que desarrollan sepsis, vale `0` antes de las seis horas previas al inicio definido y `1` desde ese momento en adelante. En pacientes sin sepsis permanece en `0`. La etiqueta anticipada define una tarea de alerta temprana; no implica que la sepsis ya estuviera presente seis horas antes. |

## Interpretación de unidades

- **mmHg:** milímetros de mercurio, unidad de presión.
- **mmol/L:** milimoles por litro, unidad de concentración de cantidad de sustancia.
- **mg/dL:** miligramos por decilitro, unidad de concentración de masa.
- **g/dL:** gramos por decilitro.
- **ng/mL:** nanogramos por mililitro.
- **IU/L:** unidades internacionales por litro; en estas enzimas expresa actividad enzimática, no masa.
- **µL:** microlitro, una millonésima parte de un litro.

### Unidades que requieren verificación

La tabla oficial publica `FiO2` en porcentaje, `Lactate` en mg/dL y `Magnesium` en mmol/dL. Estas etiquetas no bastan para confirmar la escala efectiva de cada archivo. Antes de convertir unidades o aplicar límites clínicos, contrastar la documentación de origen y los valores observados por hospital.

Para `FiO2`, una proporción de `0.21` equivale a `21 %`; no deben tratarse como escalas intercambiables sin conversión. Las unidades publicadas para lactato y magnesio también requieren confirmación: no se ha validado aquí la unidad efectiva de los archivos. No convertir automáticamente basándose solo en que una unidad parezca inusual.

## Fuente

[PhysioNet — Early Prediction of Sepsis from Clinical Data, versión 1.0.0](https://physionet.org/content/challenge-2019/1.0.0/), tabla de columnas de los archivos de entrenamiento.

DOI: https://doi.org/10.13026/v64v-d857
