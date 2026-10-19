# ♻️ Residuo Cero · clasificador de residuos con IA

App web que clasifica la foto de un residuo (cartón, vidrio, metal, papel, plástico o basura), avisa cuando la imagen no parece un residuo, muestra la confianza de cada clase y genera una recomendación de reciclaje con un modelo de lenguaje (Gemini).

**Proyecto grupal · Certamen · IA Generativa para la Gestión Industrial · UDD**

🔗 **App publicada:** https://residuocero.streamlit.app/
📦 **Repositorio:** https://github.com/El-Lags/residuo_cero

## Integrantes
- Felipe Alonso
- Sofía Fariña
- Vicente Lagunas

## Flujo
📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Resultado visual → 💬 Gemini → ✅ Recomendación

## Problema
Separar mal los residuos contamina lotes completos de material reciclable. La app ayuda a identificar el material y orienta sobre cómo y dónde reciclarlo, con aviso de revisión humana cuando el modelo no está seguro.

## Datos
- **Dataset:** TrashNet (Thung y Yang, Stanford, 2016), archivo `dataset-resized.zip` del repositorio de Hugging Face [`garythung/trashnet`](https://huggingface.co/datasets/garythung/trashnet) (2.527 fotos; el repositorio también trae la versión original de las mismas fotos, que no se usa para no duplicarlas).
- **Licencia:** MIT.
- **Clases de residuos:** cardboard, glass, metal, paper, plastic, trash.
- **Clase `other` (objetos que no son residuos):** 400 fotos para que el modelo no fuerce un material cuando la imagen es otra cosa (por ejemplo, un zapato). Vienen de dos fuentes públicas de Hugging Face: 200 de [ImageNet-1k](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256) (50 categorías cotidianas × 4 fotos: calzado, ropa, mochilas, electrónicos, muebles, animales, instrumentos, vehículos y paisajes) y 200 de [Fashion Products](https://huggingface.co/datasets/ceyda/fashion-products-small) (90 de calzado y 110 de ropa y accesorios, sobre fondo blanco, para que el modelo no use «fondo liso» como atajo). Se excluyeron las categorías que se parecen a residuos (botellas, latas, cartones, bolsas, libros, tazas).
- **Imágenes por clase (2.927 en total):** cardboard 403 · glass 501 · metal 410 · other 400 · paper 594 · plastic 482 · trash 137. La clase `trash` es solo el 4,7 % (desbalance de 4,3 a 1 entre la mayor y la menor).
- **Duplicados:** se eliminaron 3 copias exactas, quedan 2.924 imágenes.
- **División estratificada (semilla 1993):**

| Clase | Train | Val | Test |
|---|---|---|---|
| cardboard | 282 | 61 | 60 |
| glass | 350 | 75 | 76 |
| metal | 286 | 62 | 61 |
| other | 280 | 60 | 60 |
| paper | 416 | 89 | 89 |
| plastic | 336 | 72 | 72 |
| trash | 96 | 20 | 21 |
| **Total** | **2.046** | **439** | **439** |

- **Pruebas reservadas:** 30 zapatos y 40 fotos de 8 categorías de objetos que el modelo nunca vio (pizza, castillo, tractor, león, microondas, refrigerador, lavadora y aspiradora) quedaron fuera de train, val y test, para medir el problema original.
- **Limitaciones:** residuos de una sola fuente con fondos lisos, clase `trash` escasa, clase `other` armada con fotos de catálogos e ImageNet, posibles casi-duplicados. Detalle en el notebook.
- Los datasets **no** están en este repositorio por su tamaño; se descargan desde los enlaces.

## Modelo
| Modelo | Accuracy (test) | Accuracy balanceada (test) |
|---|---|---|
| CLIP zero-shot (línea base) | 50,8 % | 46,4 % |
| **EfficientNet-B0 (timm), transfer learning** | **87,9 %** | **88,1 %** |

Elegimos EfficientNet-B0 porque supera a CLIP por 37,1 puntos de accuracy y 41,7 de accuracy balanceada con las mismas 439 imágenes de prueba, y además pesa unos 16 MB, cabe en GitHub y corre en CPU. Es un modelo de **7 clases** (6 materiales y `other`).

| Clase | Recall (test) |
|---|---|
| other | 95,0 % |
| metal | 93,4 % |
| cardboard | 90,0 % |
| paper | 87,6 % |
| trash | 85,7 % |
| glass | 84,2 % |
| plastic | 80,6 % |

La clase con más errores de precisión es `trash` (precision 0,69 con solo 21 imágenes de prueba). Con la clase `other` mejoraron metal, papel, plástico y `trash`, pero **bajaron vidrio (de 96 % a 84 %) y cartón (de 98 % a 90 %)** respecto al modelo anterior de 6 clases. La comparación no es exacta, porque el test cambió (439 fotos en vez de 379) y las clases tienen pocas fotos.

**Error más costoso:** clasificar como reciclable un residuo no reciclable, porque contamina el lote completo. Ocurrió con 3 de las 21 imágenes `trash` (14,3 %). El error inverso también cuesta, porque se pierde material reciclable: 8 imágenes reciclables se clasificaron como `trash` (3 vidrio, 3 papel y 2 plástico) y 2 como `other` (1 cartón y 1 papel).

**Objetos que no son residuos (problema original):** antes, un zapato se clasificaba como vidrio con 92 % de confianza. Con la clase `other`, los 30 zapatos reservados se detectaron como `other` (100 %) y el 75 % de las fotos de categorías nunca vistas (30 de 40); los 10 fallos se fueron a metal (5), papel (2), plástico (2) y cartón (1). Solo 2 de 379 residuos reales (0,5 %) se marcaron como `other`. Los zapatos reservados vienen del mismo catálogo que el entrenamiento, así que el 100 % es una cota optimista.

## Estructura
```
app.py                       aplicación Streamlit
requirements.txt             librerías
notebooks/entrenamiento.ipynb  datos, comparación de modelos y métricas
modelo/                      pesos (.pt), clases y métricas del test
ejemplos/                    imágenes de prueba
.streamlit/config.toml       tema de la app
```

## Cómo ejecutarla
```bash
pip install -r requirements.txt
# clave de Gemini (nunca se sube a GitHub)
mkdir .streamlit   # si no existe
echo GEMINI_API_KEY = "tu_clave" > .streamlit/secrets.toml
streamlit run app.py
```
En Streamlit Cloud la clave se guarda en **Advanced settings → Secrets** como `GEMINI_API_KEY = "..."`.

## Uso de IA

- **Modelos de visión:** Se utilizó EfficientNet-B0 (`timm`), ajustado con TrashNet e incorporando la clase `other` mediante imágenes de ImageNet y Fashion Products. CLIP (OpenAI) se empleó como modelo de referencia.

- **IA generativa en la aplicación:** Google Gemini se utiliza para generar recomendaciones textuales a partir de la clase predicha y su nivel de confianza. La imagen no es enviada al modelo.

- **Apoyo durante el desarrollo:** Se utilizaron herramientas de asistencia de programación para apoyar tareas puntuales de implementación, revisión de código y documentación durante la etapa final del proyecto.

- **Entrenamiento y evaluación:** El entrenamiento de los modelos y el cálculo de las métricas se realizaron en Google Colab.

## Consideraciones éticas
- Prototipo **educativo**: no reemplaza las indicaciones de la municipalidad ni de un punto limpio.
- Con confianza menor a 70 % la app pide revisión humana. En el test, las fotos con confianza ≥ 70 % (89 % del total) tienen 93,9 % de accuracy; las demás se derivan a revisión.
- No recolectamos fotos de personas ni datos sensibles. Sin embargo, la clase `other` usa fotos públicas de ImageNet y de un catálogo de moda, donde pueden aparecer personas (por ejemplo, con la ropa puesta); se usan solo con fines educativos y esas fuentes tienen sus propios términos de uso, que conviene revisar antes de cualquier uso distinto.
- Mundo cerrado, mitigado pero no resuelto: antes el modelo no tenía una clase «otro» y clasificó un zapato negro como vidrio con 92,4 % de confianza. Ahora la clase `other` detecta los zapatos de prueba y el 75 % de las categorías nunca vistas, pero un objeto muy distinto a los del entrenamiento aún puede clasificarse como un material, incluso con confianza alta. Por eso se mantiene la revisión humana.
- Sesgo: el dataset proviene de productos de EE. UU. con fondos controlados; puede rendir peor con envases chilenos o fondos complejos.
