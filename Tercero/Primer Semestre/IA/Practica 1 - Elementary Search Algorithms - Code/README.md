README - Practica 1 - Elementary Search Algorithms - Code

Cambios por archivo y método 
-------------------------------------

1) aichess.py
  - `__init__(...)`
    - Añade estructuras de apoyo para búsqueda: `dictPath` (padres/g) y `dictVisitedStates`.
    - Por qué: permitir reconstruir rutas y llevar seguimiento fiable de visitados durante A*/DFS.

  - `newBoardSim(listStates)`
    - Refactor que reconstruye `boardSim` desde una lista de estados (lista de piezas). Usada antes de generar sucesores.
    - Por qué: evita aplicar diffs peligrosos y garantiza una simulación consistente.

  - `getWhiteState` / `getBlackState` / `getPieceState` / `getCurrentState`
    - Pequeñas clarificaciones: estas funciones normalizan y extraen la representación canonizada de piezas (p. ej. rey en índice 0) usada por la heurística y por el almacenamiento en `dictPath`.

  - `getListNextStatesW/B` (wrapper)
    - Ahora invoca la generación de sucesores sobre `boardSim` (posición reconstruida) y devuelve estados normalizados.

  - `h(state)`
    - Heurística revisada: combina distancia Manhattan del rey blanco hacia la casilla objetivo y la distancia para alinear la torre; devuelve `max(...)` para mantener admisibilidad.
    - Por qué: estimador más informativo, reduce la exploración en muchas posiciones.

  - `AStarSearch(currentState)`
    - Reescrita para usar `heapq` (frontera con tuplas `(f, state, g)`), `cost_so_far` y `dictPath` para padres. En cada expansión reconstruye `boardSim` desde el estado y las piezas negras originales, genera sucesores y actualiza la frontera.
    - Por qué: permitir reconstrucción del camino final y mantener consistencia al generar sucesores.

  - `isCheckMate(mystate)`
    - Implementación dinámica: localiza rey negro en la simulación, usa un detector puro de ataques para saber si está en jaque, y simula todas las jugadas negras (capturas/interposiciones) para verificar si alguna quita el jaque; si no existe, devuelve True (mate).
    - Por qué: reemplazar una comprobación hardcodeada por una lógica completa y segura contra efectos secundarios.

  - `list_black_legal_escapes()`
    - Nueva rutina diagnóstica que devuelve la lista de jugadas negras que anulan el jaque (útil para debugging y validar la salida de A*).

2) chess.py
  - `__init__(initboard, myinit=True)`
    - Asegura inicialización consistente de `board` y `boardSim` y estado para ghost pawns.

  - `newBoardSim(initboard)`
    - Wrapper para resetear `boardSim` con un nuevo `initboard` (usado por `aichess` al reconstruir posiciones).

  - `moveSim(start, to, verbose=True)` y `move(start, to)`
    - Correcciones de bugs en la lógica de en-passant (eliminación del ghost pawn), indexación y en la actualización de `currentStateW/B`. Se limpiaron prints que causaban excepciones fuera de scope.
    - Por qué: evitar corrupciones de estado durante simulaciones y en el tablero real.

3) piece.py
  - Validadores auxiliares (`check_diag`, `check_updown`, `check_diag_castle`, `check_updown_castle`)
    - Se suprimieron prints ruidosos y ahora retornan booleans silenciosamente.

  - `King.is_valid_move` / `Pawn.is_valid_move`
    - `King.is_valid_move` valida el patrón (un paso + enroque cuando aplica) pero la comprobación de si una casilla está atacada se realiza fuera, con el detector puro.
    - `Pawn.is_valid_move` mantiene la lógica de movimiento (incluyendo la creación de `GhostPawn` en doble avance); por seguridad, el detector puro evita llamar a `is_valid_move` cuando solo queremos chequear ataques.

4) board.py
  - `getListNextStatesW/B`
    - No se cambiaron los algoritmos centrales durante la sesión. Estas funciones siguen siendo la fuente de sucesores; `aichess` ahora se asegura de invocarlas sobre una `boardSim` reconstruida.

