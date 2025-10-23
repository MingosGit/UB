# STRICT COMPLIANCE REPORT - Assignment Requirements Analysis

## Date: October 22, 2025

---

## ASSIGNMENT REQUIREMENTS SUMMARY

The assignment asks for:

> "In all cases, please do provide a **list of the visited states** from the origin to the target and the **minimal depth necessary to reach the target**."

### Specific Exercises:

1. **Exercise 1**: Both whites and blacks use Minimax (depth 4). Run 3 times. Report white wins.
2. **Exercise 2**: Vary depth from 3 to 4 for both colors. Run each combination multiple times. Plot white win percentage for each depth. Check symmetry.
3. **Exercise 3**: Whites use Minimax (no pruning), Blacks use Alpha-Beta (depth 4). Run 3 times. Report who wins.
4. **Exercise 4**: Both use Alpha-Beta. Vary depth 1-5. Run 3 times per combination. Plot proportion of wins.
5. **Exercise 5**: Whites use Expectimax, Blacks use Alpha-Beta. Run 3 times. Plot proportion of wins.

---

## COMPLIANCE ANALYSIS

### ✅ COMPLIANT ASPECTS

#### 1. Board Configuration
- **Status**: ✅ **FULLY COMPLIANT**
- **Evidence**: Lines 2939-2943 and throughout exercise runners
```python
TA[7][0] = 2   # White Rook at [7,0]
TA[7][5] = 6   # White King at [7,5]
TA[0][7] = 8   # Black Rook at [0,7]
TA[0][5] = 12  # Black King at [0,5]
```

#### 2. Exercise 1 - Minimax vs Minimax (depth 4)
- **Status**: ✅ **FULLY COMPLIANT**
- **Implementation**: 
  - Uses `minimaxNoPruning()` function (lines 1023-1143)
  - Both players use pure Minimax without any optimization
  - Runs 3 times by default
  - Tracks and reports white wins
- **Evidence**: `run_exercise_1()` function (lines 2196-2291)

#### 3. Exercise 2 - Varying Depths
- **Status**: ✅ **FULLY COMPLIANT**
- **Implementation**:
  - Tests depths [3, 4] for both colors
  - Tests all combinations: (3,3), (3,4), (4,3), (4,4)
  - Runs 3 times per combination
  - Generates plot of white win percentage
  - Checks symmetry
- **Evidence**: `run_exercise_2()` function (lines 2294-2493)

#### 4. Exercise 3 - Minimax vs Alpha-Beta
- **Status**: ✅ **FULLY COMPLIANT**
- **Implementation**:
  - White uses `minimaxNoPruning()` (no pruning)
  - Black uses `alphabeta()` (with pruning)
  - Depth 4 for both
  - Runs 3 times
  - Reports winner for each game
- **Evidence**: 
  - `run_exercise_3()` function (lines 2495-2596)
  - `alphaBetaGame()` correctly handles both algorithms (lines 1750-1764)

#### 5. Exercise 4 - Alpha-Beta vs Alpha-Beta
- **Status**: ✅ **FULLY COMPLIANT**
- **Implementation**:
  - Both players use `alphabeta()` with pruning
  - Varies depth from 1 to 5 for both
  - Runs 3 times per combination (25 combinations × 3 = 75 games)
  - Generates plot showing win proportions
- **Evidence**: 
  - `run_exercise_4()` function (lines 2599-2780)
  - Uses `alphaBetaGame()` with both flags set to True (line 2668)

#### 6. Exercise 5 - Expectimax vs Alpha-Beta
- **Status**: ✅ **FULLY COMPLIANT**
- **Implementation**:
  - White uses `expectimaxValue()` algorithm
  - Black uses `alphabeta()` with pruning
  - Depth 4 for both
  - Runs 3 times
  - Generates plot with win proportions
- **Evidence**: 
  - `run_exercise_5()` function (lines 2783-2928)
  - `expectimaxGame()` function (lines 1984-2187)

#### 7. Algorithm Implementations
- **Status**: ✅ **ALL ALGORITHMS CORRECTLY IMPLEMENTED**
- **Minimax (no pruning)**: Lines 1023-1143 - Pure minimax, NO transposition table, NO cache
- **Alpha-Beta pruning**: Lines 1146-1324 - WITH transposition table and optimizations
- **Expectimax**: Lines 1889-1981 - Proper chance nodes for black

---

### ⚠️ PARTIAL COMPLIANCE / ISSUES

#### 1. **CRITICAL ISSUE**: "Minimal Depth Necessary to Reach the Target"
- **Status**: ❌ **NOT FULLY COMPLIANT**
- **Problem**: The code does NOT track or report the "minimal depth necessary to reach the target"
- **Evidence**: Line 1364-1366 explicitly states this limitation:
```python
# NOTE: Minimum depth calculation would require tracking actual search depth used
# This is a known limitation - we report configured depth instead
# To implement properly would require modifying all search algorithms to return depth info
```
- **What is reported**: The CONFIGURED search depth (e.g., depth=4)
- **What is REQUIRED**: The ACTUAL depth at which checkmate/terminal state was found
- **Explanation**: The assignment asks for "minimal depth necessary to reach the target" which means:
  - If configured depth is 4, but checkmate occurs at depth 2, report 2 (not 4)
  - This requires the search algorithms to return depth information
  - Current implementation does NOT track this

#### 2. **"List of Visited States"**
- **Status**: ⚠️ **AMBIGUOUS - PARTIALLY COMPLIANT**
- **What is PROVIDED**: 
  - The code tracks `visitedStates` which contains the SEQUENCE of board states in the actual game
  - This is saved to files (e.g., `states_ex1_1.txt`)
  - Example: Initial state → White move 1 → Black move 1 → White move 2 → ... → Terminal state
- **What MIGHT BE REQUIRED**: 
  - The assignment is unclear whether it wants:
    - A) The sequence of states in the GAME (what you provide) ✅
    - B) ALL states EXPLORED during search (much larger - all nodes visited in the tree) ❌
- **Evidence**: 
  - Game states ARE tracked: Lines 1353-1354, 1414-1416, 1693-1694
  - These are saved via `_save_game_data()`: Lines 1004-1015
  - Search tree exploration is NOT tracked (no record of all minimax/alphabeta nodes visited)

#### 3. **Exercise 2 Plot Requirement**
- **Status**: ⚠️ **INTERPRETATION ISSUE**
- **Assignment asks**: "Plot the percentage of white wins over the total for each depth value"
- **What you plot**: White win percentage averaged across all opponent depths for each white depth
- **Possible interpretation**: Should you plot separately for each (white_depth, black_depth) combination?
- **Current implementation**: Averages across black depths (lines 2437-2441)
- **Assessment**: Your interpretation is REASONABLE but could be more explicit

---

## DETAILED ISSUES AND RECOMMENDATIONS

### ISSUE #1: Minimal Depth Tracking (CRITICAL)

**Problem**: The assignment explicitly requires reporting "the minimal depth necessary to reach the target" but your code only reports the configured search depth.

**Why this matters**: 
- If you configure depth=4 but checkmate happens at move 2 (depth 2), you should report depth=2
- This measures EFFICIENCY of the algorithm

**How to fix**:
1. Modify search algorithms to return depth information:
```python
def minimaxNoPruning(self, state, depth, isWhite, positionHistory=None, initial_depth=None):
    if initial_depth is None:
        initial_depth = depth
    # ... existing code ...
    # At terminal state detection:
    actual_depth_used = initial_depth - depth
    return (value, state, actual_depth_used)
```

2. Track in game loop:
```python
# In minimaxGame, alphaBetaGame, etc.
value, bestState, depth_used = self.minimaxNoPruning(...)
minDepthWhite = min(minDepthWhite, depth_used)
```

3. Report in results:
```python
'stats': {
    'configured_depth_white': depthWhite,
    'min_depth_used_white': minDepthWhite,  # This is what's MISSING
    # ...
}
```

**Severity**: HIGH - This is explicitly mentioned in the assignment requirements

---

### ISSUE #2: Visited States Clarification

**Ambiguity**: "List of visited states" could mean:
1. States in the ACTUAL GAME PATH (what you provide) ✅
2. ALL states EXPLORED during search (what's likely intended for search algorithm analysis) ❌

**Current implementation**: Only tracks game path states, not search tree exploration

**If interpretation #2 is correct**, you need to:
1. Add a counter in search algorithms:
```python
self.states_explored = 0

def minimaxNoPruning(...):
    self.states_explored += 1
    # ... rest of code
```

2. Report in results:
```python
'stats': {
    'states_in_game_path': len(visitedStates),
    'states_explored_by_minimax': self.states_explored,  # MISSING
}
```

**Recommendation**: Check with professor which interpretation is correct. For AI algorithm analysis, typically we care about nodes explored in the search tree.

---

### ISSUE #3: Exercise 2 Symmetry Check

**Current implementation**: Computes symmetry but could be more explicit

**From your code** (lines 2471-2492):
```python
# Symmetry analysis (only if we have at least 2 depths)
if len(depth_values) >= 2:
    print("\n" + "="*70)
    print("SYMMETRY ANALYSIS")
    # ...
```

**Issue**: The symmetry check is implemented BUT NOT VISIBLE in the provided code snippet. Need to verify this section is complete.

**What symmetry means**: 
- (3,4) white wins should roughly equal (4,3) black wins
- Because swapping depths should swap advantages

**Recommendation**: Ensure you explicitly compute and report symmetry metrics

---

## SUMMARY TABLE

| Requirement | Status | Compliance Level |
|------------|--------|------------------|
| Board configuration correct | ✅ | 100% |
| Exercise 1: Minimax vs Minimax | ✅ | 100% |
| Exercise 2: Varying depths | ✅ | 95% (symmetry check unclear) |
| Exercise 3: Minimax vs Alpha-Beta | ✅ | 100% |
| Exercise 4: Alpha-Beta vs Alpha-Beta | ✅ | 100% |
| Exercise 5: Expectimax vs Alpha-Beta | ✅ | 100% |
| Algorithm implementations | ✅ | 100% |
| List of visited states | ⚠️ | 50% (ambiguous requirement) |
| Minimal depth tracking | ❌ | 0% (explicitly NOT implemented) |

---

## OVERALL ASSESSMENT

### Strengths:
1. ✅ All 5 exercises are implemented correctly
2. ✅ All 3 algorithms (Minimax, Alpha-Beta, Expectimax) are correctly implemented
3. ✅ Proper separation between Minimax (no pruning) and Alpha-Beta (with pruning)
4. ✅ Saves moves and states to files
5. ✅ Generates required plots
6. ✅ Runs correct number of repetitions
7. ✅ Board configuration matches requirements exactly

### Critical Deficiencies:
1. ❌ **Does NOT track "minimal depth necessary to reach target"** (explicitly acknowledged in line 1364-1366)
2. ⚠️ **Unclear if "visited states" means game path or search tree exploration**

### Severity Rating:
- **Exercise completion**: 95% ✅
- **Requirement compliance**: 70% ⚠️ (missing depth tracking is critical)

---

## RECOMMENDATIONS

### Priority 1: FIX MINIMAL DEPTH TRACKING
This is explicitly required by the assignment. The code comment at line 1364 acknowledges this is missing.

### Priority 2: CLARIFY VISITED STATES
Ask professor if "visited states" means:
- Game path states (currently provided)
- Search tree exploration states (not provided)

### Priority 3: VERIFY SYMMETRY ANALYSIS
Ensure Exercise 2 symmetry check is complete and reported clearly.

### Priority 4: DOCUMENT LIMITATIONS
If you cannot fix Priority 1 before submission, explicitly document this limitation in your report.

---

## CONCLUSION

Your implementation is **FUNCTIONALLY EXCELLENT** - all exercises work correctly and algorithms are properly implemented. However, there is a **CRITICAL COMPLIANCE GAP** regarding the "minimal depth" requirement that is explicitly mentioned in the assignment and explicitly acknowledged as unimplemented in your code (line 1364-1366).

**Recommendation**: Either implement depth tracking or explicitly discuss this limitation with the professor/in your report, as it's a direct requirement in the assignment text.
