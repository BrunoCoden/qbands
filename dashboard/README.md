# Monitor de tablaQ.csv

Página estática para visualizar y filtrar la información que genera 	ablaQ.csv.

## Estructura

- index.html: layout principal con filtros, resumen y tabla.
- styles.css: estilos (tema oscuro claro-compatible).
- pp.js: lógica que lee el CSV y refresca la vista automáticamente.

## Uso rápido

1. Asegurate de que el script principal esté actualizando 	ablaQ.csv.
2. Desde la raíz del repo levantá un servidor estático, por ejemplo:

   `ash
   cd c:\Users\Bruno\Documents\qbands-local
   python -m http.server 9000
   `

3. Abrí <http://localhost:9000/dashboard/> en el navegador.

La app recarga el CSV cada 5 segundos (configurable con el toggle “Auto refresh”). Si no querés auto refresco, desmarcá la casilla o hacé clic en “Actualizar ahora”.

## Filtros

- TouchUpperQ / TouchLowerQ: mostrar únicamente las filas donde hay toque.
- Últimas filas a mostrar: trunca la vista a N filas (máx. 1000) para que la tabla sea manejable.

## Notas

- Se consulta ../tablaQ.csv, por lo que la carpeta dashboard/ debe servirse desde la raíz del proyecto.
- El fetch se hace con cache: 'no-store' para evitar datos stale, pero puede seguir dependiendo del cache del navegador si se abre con ile://. Recomendado usar un servidor local.