# OCR para Cartas de Restaurantes

## Introducción
Este proyecto tiene como objetivo facilitar la digitalización de cartas de restaurantes mediante el uso de reconocimiento óptico de caracteres (OCR). Utilizando un modelo de detección y segmentación basado en YOLOv8, el programa identifica secciones individuales en la carta, tales como nombres de platos, descripciones y precios. Una vez segmentado, el texto extraído es procesado y exportado en un formato estructurado a un archivo de texto (.txt), lo cual permite su fácil manipulación y almacenamiento.

## Fine-Tuning de YOLOv8

Para adaptar YOLOv8 a la detección y segmentación de componentes específicos en cartas de restaurantes, se ha realizado un ajuste fino (fine-tuning) del modelo utilizando un conjunto de datos personalizado. Este dataset incluye imágenes de menús de diferentes idiomas, tamaños y disposiciones, proporcionando una mayor versatilidad y precisión en la detección. Las etiquetas para la segmentación incluyen:

- `0` - Descripción del plato: detalla los ingredientes o características.
- `1` - Área de todos los componentes del plato: incluye todas las secciones visibles del plato.
- `2` - Precio del plato: permite extraer el costo asociado.
- `3` - Título del plato: contiene el nombre o título principal del plato.

Estas etiquetas facilitan una detección precisa y organizada de los elementos más relevantes en una carta de restaurante.

### Aumentación de Datos con Albumentations

Para mejorar la robustez del modelo y su capacidad de generalización, se ha aplicado un conjunto de transformaciones de datos usando la librería **Albumentations**. Estas transformaciones incluyen:

- Conversión a escala de grises.
- Inversión horizontal aleatoria.
- Ajuste de brillo y contraste aleatorio.
- Desplazamiento, escalado y rotación para diferentes perspectivas.
- Aplicación de desenfoque leve para simular distintos niveles de calidad de imagen.

Estas aumentaciones ayudan al modelo a manejar variaciones en las cartas, como diferentes resoluciones, iluminaciones y orientaciones, logrando así un mejor rendimiento en situaciones reales.

## Asignación de Componentes a Cada Elemento del Menú

Este proyecto utiliza un modelo YOLOv8 entrenado específicamente para segmentar y clasificar elementos en cartas de restaurantes. La segmentación incluye identificar áreas de cada plato en la carta, sus títulos, descripciones y precios. La siguiente fase del procesamiento organiza estos componentes, asignándolos a cada elemento del menú detectado.

### Extracción de Bounding Boxes
El modelo YOLOv8, entrenado en cartas de menú, permite extraer *bounding boxes* que rodean cada componente identificado en la imagen. Para cada imagen procesada:

1. Se obtienen las coordenadas de cada *bounding box* junto con su clase y confianza.
2. Si se habilita, el programa también dibuja estas cajas en la imagen, creando una visualización útil para verificar la precisión del modelo.

### Organización de Componentes por Elemento de Menú
Cada elemento principal del menú (área completa de un plato) se analiza para detectar componentes secundarios que le pertenezcan, como el título, descripción y precio. Para ello:

1. El modelo clasifica los bounding boxes en categorías de `título`, `descripción` y `precio` según sus etiquetas.
2. Utilizando una función de contención, se verifica si estos componentes secundarios están completamente contenidos dentro del área de cada elemento del menú.
3. Finalmente, se organiza la información en un diccionario que asocia cada elemento del menú con sus componentes correspondientes.




