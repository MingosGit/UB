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

Notas sobre rendimiento y consistencia
------------------------------------
- Reconstruir `boardSim` en cada expansión es seguro y sencillo, pero más lento que aplicar diffs. Si la búsqueda crece mucho, conviene implementar una sincronización incremental robusta (aplicar y deshacer moves) en vez de reconstrucciones completas.
- He dejado varios prints de depuración en el código para que puedas ver el proceso de A*. Si prefieres que los quite o que los haga opcionales con un flag `verbose`, dímelo.

Cambios menores recientes
------------------------
Se han eliminado mensajes ruidosos del archivo `piece.py` (por ejemplo: "This piece does not move in this pattern.") que imprimían durante la comprobación de patrones de movimiento. Estos mensajes se reemplazaron por retornos silenciosos para evitar saturar la salida durante la búsqueda y el diagnóstico. La lógica de validación de movimientos no ha cambiado; sólo se suprimió el logging intrusivo. Si prefieres mantener estos mensajes en modo debug, puedo modificar el código para que sean condicionales con un flag `verbose`.

Cambios técnicos recientes (detallado)
-----------------------------------
Además de la supresión de mensajes ruidosos, se añadieron y/o mejoraron las siguientes funcionalidades para garantizar resultados correctos y facilitar el diagnóstico:

- isCheckMate dinámico: El detector ya no usa listas hardcodeadas; en su lugar inspecciona la posición en `boardSim`, detecta si el rey negro está en jaque y simula todas las jugadas legales negras (incluyendo capturas) para comprobar si alguna elimina el jaque. Si ninguna lo hace, se considera mate.

- Detector puro de ataques: Para evitar efectos secundarios (por ejemplo, `Pawn.is_valid_move` que crea un `GhostPawn` al validar), la detección de ataques se realiza con una función pura que calcula si una casilla está atacada por blancas sin modificar `boardSim`.

- Diagnóstico de escapes negros: Cuando A* detecta mate, se ejecuta una rutina diagnóstica que lista todas las jugadas negras que, de existir, eliminarían el jaque. Esto facilita verificar casos en los que el resultado pueda parecer incorrecto.

- Heurística mejorada en A*: La función heurística `h(state)` ahora combina la distancia Manhattan del rey blanco a la casilla objetivo y la distancia mínima necesaria para que la torre se alinee con esa casilla; se usa el máximo de estas dos medidas para un estimador admisible más ajustado (reduce nodos explorados en muchos casos).

Todos estos cambios fueron implementados con cuidado para mantener la semántica original del juego y facilitar la depuración; si quieres que documente ejemplos de salida o que agregue tests unitarios que validen las propiedades (admisibilidad, detección de mate), puedo agregarlos a continuación.

Archivos modificados (resumen por fichero)
-----------------------------------------
- `aichess.py`
  - A*: añadido frontier (heapq), `cost_so_far` y `dictPath` para reconstruir la ruta final.
  - Normalización de estados blancos (`getWhiteState`) para consistencia en heurística y almacenamiento.
  - `isCheckMate`: reemplazado por una versión dinámica que detecta jaque, usa un detector puro de ataques y simula todas las jugadas negras para confirmar mate.
  - Añadida función diagnóstica `list_black_legal_escapes()` que lista jugadas negras que eliminarían el jaque si existen.
  - Heurística `h(state)` mejorada (max(kingDistance, rookAlignDist)).

- `chess.py`
  - Correcciones en `move`/`moveSim` (en-passant, indexación de ghost pawns) y limpieza de prints problemáticos.
  - Asegurada la consistencia entre `board` y `boardSim` en actualizaciones de estado.

- `piece.py`
  - Suprimidos prints ruidosos (p.ej. "This piece does not move in this pattern.") reemplazándolos por retornos silenciosos o comentarios; la lógica de validación no cambió.

- `board.py`
  - Sin cambios funcionales significativos; sirve como generador de sucesores para A* y DFS/BFS.

Si quieres, puedo añadir un pequeño ejemplo de ejecución (salida típica) o crear pruebas unitarias que comprueben los casos de mate/no-mate automáticamente.
