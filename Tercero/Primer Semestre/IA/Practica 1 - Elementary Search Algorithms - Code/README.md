README - Practica 1 - Elementary Search Algorithms - Code

Descripción
-----------
Este repositorio contiene una versión pequeña de un motor/AI para el final rey+torre vs rey, junto con utilidades de tablero y piezas.

Archivos relevantes
------------------
- `aichess.py`: Clase `Aichess` que ejecuta búsquedas (DFS, BFS, A*). Aquí se realizaron los cambios principales para que A* reconozca y reconstruya la ruta hacia mate.
- `chess.py`: Clase `Chess` con funciones `move` y `moveSim`. Se limpiaron errores que impedían la correcta simulación y actualización del estado AI.
- `board.py`: Clase `Board` que genera sucesores (sin cambios significativos hechos por mí, pero es importante para entender la representación de estados).
- `piece.py`: Definición de las piezas y comprobaciones de movimiento.

Cambios aplicados (resumen detallado)
-------------------------------------
He realizado cambios mínimos y específicos para que la búsqueda A* funcione y para eliminar errores que rompían la simulación. Los cambios clave son:

1) aichess.py (A* y sincronización)
- A* ahora mantiene g-scores y padres (mapa `dictPath`) para poder reconstruir la ruta final.
  - Antes: A* no almacenaba parents/g correctamente, por lo que `pathToTarget` quedaba vacío.
  - Ahora: se usa un `frontier` (heapq), `cost_so_far` (g) y `dictPath[normalized_state] = (parent_state, g)`.
- Normalización de estados blancos: antes de encolar un estado, se normaliza con `getWhiteState` para que la heurística y el almacenamiento de padres usen siempre la misma representación (lista `[rey, torre]`).
- Reconstrucción de `boardSim` cuando A* expande un nodo:
  - En vez de intentar aplicar moves incrementalmente con lógica frágil, el código reconstruye `boardSim` a partir del estado blanco normalizado del nodo + las piezas negras actuales (desde `self.chess.board.currentStateB`). Esto evita inconsistencias al generar sucesores.
- Se añadió la llamada a `reconstructPath` cuando A* encuentra el objetivo para rellenar `pathToTarget`.

2) chess.py (move & moveSim corrections)
- Correcciones de en-passant: índices usados para eliminar el "ghost pawn" estaban equivocados; se corrigió la indexación para eliminar el fantasma blanco/negro correctamente.
- Eliminados/ajustados prints que referenciaban `m` fuera de su scope. Esto evitó excepciones en tiempo de ejecución.
- Arreglada la mezcla `board` vs `boardSim` en varios lugares: las actualizaciones del estado AI durante simulaciones ahora usan `boardSim.currentStateW` y `boardSim.currentStateB`.
- Corregida indentación y comparaciones para que la actualización de `currentStateW`/`currentStateB` sea consistente y no falle con errores de compilación.

3) Otros
- No se cambiaron los algoritmos de `Board.getListNextStatesW/B` salvo una limpieza menor en `chess.py` para evitar debug prints problemáticos. La lógica de sucesores sigue siendo la del código original.

Notas de ejecución
------------------
- Cómo ejecutar el ejemplo usado para pruebas:

  1) Abre un terminal en Windows (PowerShell). Asegúrate de usar el Python adecuado (en mi entorno ejecuté con Anaconda).
  2) Ejecuta:

```powershell
& C:/Users/josea/anaconda3/python.exe "c:/Users/josea/Desktop/Github/UB/Tercero/Primer Semestre/IA/Practica 1 - Elementary Search Algorithms - Code/aichess.py"
```

- Resultado esperado (según la posición inicial en el script): A* explorará estados y, si encuentra mate dentro de los límites, imprimirá "Checkmate found!" y un `pathToTarget` con la secuencia de estados normalizados desde la situación inicial hasta la mate.

Notas sobre rendimiento y consistencia
------------------------------------
- Reconstruir `boardSim` en cada expansión es seguro y sencillo, pero más lento que aplicar diffs. Si la búsqueda crece mucho, conviene implementar una sincronización incremental robusta (aplicar y deshacer moves) en vez de reconstrucciones completas.
- He dejado varios prints de depuración en el código para que puedas ver el proceso de A*. Si prefieres que los quite o que los haga opcionales con un flag `verbose`, dímelo.

Siguientes pasos (opcional)
--------------------------
- Añadir pruebas unitarias (pytest) que validen la ruta para esta posición inicial.
- Mejorar la heurística o hacer que A* limite profundidad/tiempo.
- Implementar una sincronización incremental segura para `boardSim` (más eficiente).

Contacto
--------
Si quieres que haga alguno de los pasos opcionales, dime cuál y lo implemento.
