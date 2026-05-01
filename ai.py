"""
╔══════════════════════════════════════════════════════════════════════╗
║        WUMPUS WORLD — KNOWLEDGE-BASED LOGIC AGENT                   ║
║        Propositional Logic + CNF Resolution Refutation               ║
║        NUCES Chiniot-Faisalabad | AI Assignment Q6                   ║
╚══════════════════════════════════════════════════════════════════════╝

Features:
  • Dynamic grid sizing (user-defined Rows × Cols)
  • Random Pit & Wumpus placement each episode
  • Percept generation: Breeze (adj pit), Stench (adj wumpus), Glitter
  • Propositional Logic Knowledge Base (CNF clause storage)
  • TELL: biconditional rules → CNF encoding
  • ASK: Resolution Refutation (proof by contradiction on CNF)
  • BFS navigation through proven-safe cells
  • Full colour terminal display + real-time metrics dashboard
"""

import random
import os
import sys
import time
from collections import deque
from copy import deepcopy

# ─── ANSI Colours ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    PURPLE = "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"
    GRAY   = "\033[90m"
    BG_RED    = "\033[41m"
    BG_GREEN  = "\033[42m"
    BG_BLUE   = "\033[44m"
    BG_YELLOW = "\033[43m"
    BG_GRAY   = "\033[100m"

def clr(text, *codes):
    return "".join(codes) + str(text) + C.RESET

# ─── Literal & Clause Utilities ───────────────────────────────────────────────
class Literal:
    """A propositional literal: a variable name + negation flag."""
    def __init__(self, var: str, neg: bool = False):
        self.var = var
        self.neg = neg

    def negate(self):
        return Literal(self.var, not self.neg)

    def __eq__(self, other):
        return self.var == other.var and self.neg == other.neg

    def __hash__(self):
        return hash((self.var, self.neg))

    def __repr__(self):
        return f"{'¬' if self.neg else ''}{self.var}"

    def clause_key(self):
        return ('~' if self.neg else '') + self.var


def clause_key(clause: list) -> frozenset:
    """Canonical key for a clause (set of literals)."""
    return frozenset(lit.clause_key() for lit in clause)


def is_tautology(clause: list) -> bool:
    """True if clause contains both x and ¬x."""
    vars_pos = {l.var for l in clause if not l.neg}
    vars_neg = {l.var for l in clause if l.neg}
    return bool(vars_pos & vars_neg)


def deduplicate(clause: list) -> list:
    seen = set()
    result = []
    for l in clause:
        k = l.clause_key()
        if k not in seen:
            seen.add(k)
            result.append(l)
    return result


# ─── Knowledge Base ───────────────────────────────────────────────────────────
class KnowledgeBase:
    """
    Propositional Logic Knowledge Base stored in CNF.
    Supports TELL (add clauses) and ASK (resolution refutation).
    """

    def __init__(self):
        self.clauses: list[list[Literal]] = []
        self._clause_keys: set[frozenset] = set()
        self.inference_steps = 0

    def tell(self, clause: list[Literal], source: str = "") -> bool:
        """Add a CNF clause to the KB. Returns True if it was new."""
        clause = deduplicate(clause)
        if is_tautology(clause):
            return False
        ck = clause_key(clause)
        if ck in self._clause_keys:
            return False
        self.clauses.append(clause)
        self._clause_keys.add(ck)
        return True

    def tell_all(self, clauses: list[list[Literal]]):
        for cl in clauses:
            self.tell(cl)

    def ask(self, goal: Literal) -> bool:
        """
        ASK: Is `goal` entailed by the KB?
        Method: Resolution Refutation.
          1. Add ¬goal as a unit clause.
          2. Resolve pairs of clauses.
          3. If empty clause derived → contradiction → goal proved.
          4. If no new clauses generated → goal not provable.
        """
        self.inference_steps += 1
        neg_goal = goal.negate()

        # Working set starts with KB + {¬goal}
        working: list[list[Literal]] = [list(c) for c in self.clauses]
        working.append([neg_goal])
        seen_keys: set[frozenset] = set(clause_key(c) for c in working)

        max_iterations = 600
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            new_clauses = []
            n = len(working)

            for i in range(n):
                for j in range(i + 1, n):
                    resolvents = self._resolve_pair(working[i], working[j])
                    for r in resolvents:
                        if len(r) == 0:
                            return True          # Empty clause → proved
                        ck = clause_key(r)
                        if ck not in seen_keys:
                            seen_keys.add(ck)
                            new_clauses.append(r)

            if not new_clauses:
                return False                     # Fixpoint — not provable
            working.extend(new_clauses)

        return False  # Timeout

    def _resolve_pair(self, A: list[Literal], B: list[Literal]) -> list[list[Literal]]:
        """Resolve clauses A and B on every complementary literal pair."""
        results = []
        for la in A:
            for lb in B:
                if la.var == lb.var and la.neg != lb.neg:
                    new_clause = [l for l in A if l != la] + [l for l in B if l != lb]
                    new_clause = deduplicate(new_clause)
                    if not is_tautology(new_clause):
                        results.append(new_clause)
        return results

    def clause_count(self) -> int:
        return len(self.clauses)

    def recent_clauses(self, n: int = 12) -> list[str]:
        return [" ∨ ".join(str(l) for l in c) for c in self.clauses[-n:]]


# ─── World ─────────────────────────────────────────────────────────────────────
class Cell:
    def __init__(self):
        self.pit = False
        self.wumpus = False
        self.gold = False

    def __repr__(self):
        parts = []
        if self.pit:    parts.append("PIT")
        if self.wumpus: parts.append("WUMPUS")
        if self.gold:   parts.append("GOLD")
        return ",".join(parts) if parts else "EMPTY"


class WumpusWorld:
    """
    Wumpus World environment.
    Generates grid with random pits, one wumpus, one gold.
    Provides percept generation.
    """

    def __init__(self, rows: int, cols: int, num_pits: int):
        self.rows = rows
        self.cols = cols
        self.num_pits = num_pits
        self.grid: dict[tuple, Cell] = {}
        self._init_grid()
        self._place_hazards()

    def _init_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[(r, c)] = Cell()

    def _place_hazards(self):
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if not (r == 0 and c == 0)]

        # Place Wumpus
        wpos = random.choice(all_cells)
        self.grid[wpos].wumpus = True
        self.wumpus_pos = wpos

        # Place Gold
        remaining = [c for c in all_cells if c != wpos]
        gpos = random.choice(remaining)
        self.grid[gpos].gold = True
        self.gold_pos = gpos

        # Place Pits
        pit_candidates = [c for c in remaining if c != gpos]
        n = min(self.num_pits, len(pit_candidates))
        pit_cells = random.sample(pit_candidates, n)
        for p in pit_cells:
            self.grid[p].pit = True
        self.pit_positions = set(pit_cells)

    def adjacent(self, r: int, c: int) -> list[tuple]:
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result.append((nr, nc))
        return result

    def get_percepts(self, r: int, c: int, wumpus_alive: bool) -> dict:
        adj = self.adjacent(r, c)
        breeze  = any(self.grid[a].pit for a in adj)
        stench  = wumpus_alive and any(self.grid[a].wumpus for a in adj)
        glitter = self.grid[(r, c)].gold
        bump    = False  # Not used in this simplified model
        scream  = False
        return {
            "breeze":  breeze,
            "stench":  stench,
            "glitter": glitter,
            "bump":    bump,
            "scream":  scream,
        }

    def is_fatal(self, r: int, c: int, wumpus_alive: bool) -> tuple[bool, str]:
        cell = self.grid[(r, c)]
        if cell.pit:
            return True, "pit"
        if cell.wumpus and wumpus_alive:
            return True, "wumpus"
        return False, ""


# ─── Agent ────────────────────────────────────────────────────────────────────
class LogicAgent:
    """
    Knowledge-Based Agent using Propositional Logic.

    Decision cycle:
      1. PERCEIVE current cell → get percepts
      2. TELL KB new biconditional rules in CNF
      3. For each unvisited adjacent cell, ASK KB if it's safe
      4. Navigate to nearest proven-safe unvisited cell via BFS
      5. Repeat until gold found, death, or no moves
    """

    def __init__(self, world: WumpusWorld):
        self.world = world
        self.kb = KnowledgeBase()
        self.pos = (0, 0)
        self.alive = True
        self.won = False
        self.wumpus_alive = True
        self.has_arrow = True

        self.visited:   set[tuple] = set()
        self.safe_set:  set[tuple] = set()
        self.danger_set: set[tuple] = set()
        self.move_count = 0
        self.log: list[tuple[str,str]] = []   # (message, colour_code)

        # Mark start as safe
        self.safe_set.add((0, 0))
        self.kb.tell([Literal(f"P_0_0", neg=True)])
        self.kb.tell([Literal(f"W_0_0", neg=True)])

    # ── Logging ────────────────────────────────────────────────────────────
    def _log(self, msg: str, colour: str = C.GRAY):
        self.log.append((msg, colour))

    # ── TELL: encode percepts as CNF biconditionals ────────────────────────
    def tell_percepts(self, r: int, c: int, percepts: dict):
        adj = self.world.adjacent(r, c)
        bvar = f"B_{r}_{c}"
        svar = f"S_{r}_{c}"

        # ── BREEZE biconditional ──────────────────────────────────────────
        # B_r_c ↔ ∨{P_ar_ac | (ar,ac) adj}
        # Split into:
        #   Forward:  B_r_c → ∨P_adj  ⟹ one clause {¬B ∨ P_a1 ∨ P_a2 ...}
        #   Backward: P_adj → B_r_c   ⟹ per adj: {¬P_adj ∨ B}
        # Plus unit facts about B itself.
        pit_lits = [Literal(f"P_{ar}_{ac}") for (ar, ac) in adj]

        if percepts["breeze"]:
            self.kb.tell([Literal(bvar)])                          # B is true
            if pit_lits:
                self.kb.tell([Literal(bvar, neg=True)] + pit_lits) # ¬B ∨ P_a1 ∨ ...
            for pl in pit_lits:
                self.kb.tell([pl.negate(), Literal(bvar)])         # ¬P_adj ∨ B
            self._log(f"  TELL: {bvar} = TRUE  (breeze at ({r},{c}))", C.CYAN)
        else:
            self.kb.tell([Literal(bvar, neg=True)])                # ¬B
            # ¬B → ¬P for each adj (via contrapositive)
            for pl in pit_lits:
                self.kb.tell([pl.negate()])                        # ¬P_adj
                self.safe_set.add((adj[pit_lits.index(pl)]))
            self._log(f"  TELL: {bvar} = FALSE (no breeze at ({r},{c}))", C.CYAN)
            self._log(f"    → All {len(adj)} adjacent cells are pit-free", C.GREEN)

        # ── STENCH biconditional ──────────────────────────────────────────
        wump_lits = [Literal(f"W_{ar}_{ac}") for (ar, ac) in adj]

        if percepts["stench"]:
            self.kb.tell([Literal(svar)])
            if wump_lits:
                self.kb.tell([Literal(svar, neg=True)] + wump_lits)
            for wl in wump_lits:
                self.kb.tell([wl.negate(), Literal(svar)])
            self._log(f"  TELL: {svar} = TRUE  (stench at ({r},{c}))", C.PURPLE)
        else:
            self.kb.tell([Literal(svar, neg=True)])
            for i, wl in enumerate(wump_lits):
                self.kb.tell([wl.negate()])
                self.safe_set.add(adj[i])
            self._log(f"  TELL: {svar} = FALSE (no stench at ({r},{c}))", C.PURPLE)
            self._log(f"    → All {len(adj)} adjacent cells are wumpus-free", C.GREEN)

        # ── Current cell is definitely safe ──────────────────────────────
        self.kb.tell([Literal(f"P_{r}_{c}", neg=True)])
        self.kb.tell([Literal(f"W_{r}_{c}", neg=True)])
        self.safe_set.add((r, c))

        # ── Uniqueness: at most one wumpus ────────────────────────────────
        # (Simplified: we add ¬W for all cells once wumpus is located)

    # ── ASK: resolution refutation ─────────────────────────────────────────
    def ask_safe(self, r: int, c: int) -> bool | None:
        """
        Returns True if cell (r,c) is proven safe,
                False if proven dangerous,
                None if unknown.
        """
        k = (r, c)
        if k in self.safe_set:  return True
        if k in self.danger_set: return False

        # Prove ¬P_r_c
        no_pit   = self.kb.ask(Literal(f"P_{r}_{c}", neg=True))
        # Prove ¬W_r_c
        no_wump  = self.kb.ask(Literal(f"W_{r}_{c}", neg=True))

        self._log(
            f"  ASK: ¬P_{r}_{c} → {'PROVED' if no_pit else 'unknown'}  "
            f"| ¬W_{r}_{c} → {'PROVED' if no_wump else 'unknown'}",
            C.YELLOW
        )

        if no_pit and no_wump:
            self.safe_set.add(k)
            return True

        # Check if definitively dangerous
        has_pit  = self.kb.ask(Literal(f"P_{r}_{c}"))
        has_wump = self.kb.ask(Literal(f"W_{r}_{c}"))
        if has_pit or has_wump:
            self.danger_set.add(k)
            return False

        return None  # Unknown

    # ── BFS planner: find nearest safe unvisited cell ─────────────────────
    def plan_next_move(self) -> tuple | None:
        start = self.pos
        queue = deque([(start, [])])
        seen = {start}

        while queue:
            (r, c), path = queue.popleft()
            for (nr, nc) in self.world.adjacent(r, c):
                if (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                safety = self.ask_safe(nr, nc)
                if safety is True and (nr, nc) not in self.visited:
                    return (nr, nc)
                if safety is True or (nr, nc) in self.visited:
                    # Can traverse; continue BFS
                    queue.append(((nr, nc), path + [(nr, nc)]))

        return None  # No safe move found

    # ── Single agent step ─────────────────────────────────────────────────
    def step(self) -> str:
        if not self.alive:
            return "dead"
        if self.won:
            return "won"

        r, c = self.pos

        # 1. Perceive
        percepts = self.world.get_percepts(r, c, self.wumpus_alive)
        self.visited.add((r, c))
        self._log(f"[Step] At ({r},{c}) — percepts: {self._fmt_percepts(percepts)}", C.WHITE)

        # 2. Tell KB
        self.tell_percepts(r, c, percepts)

        # 3. Check glitter
        if percepts["glitter"]:
            self._log("  ★ GOLD GRABBED! Mission complete!", C.YELLOW)
            self.won = True
            return "won"

        # 4. Check death
        fatal, cause = self.world.is_fatal(r, c, self.wumpus_alive)
        if fatal:
            self._log(f"  ☠ Agent died: fell into a {cause}!", C.RED)
            self.alive = False
            return "dead"

        # 5. Plan move
        target = self.plan_next_move()
        if target is None:
            # No proven-safe move — try risky adjacent unvisited cell
            adj_unvisited = [a for a in self.world.adjacent(r, c) if a not in self.visited]
            if adj_unvisited:
                target = random.choice(adj_unvisited)
                self._log(f"  ⚠ No safe move found. Taking risk: → {target}", C.RED)
            else:
                self._log("  ✗ Agent is completely stuck. No moves available.", C.RED)
                return "stuck"

        # 6. Move
        self.pos = target
        self.move_count += 1
        self._log(f"  → MOVE to {target}", C.PURPLE)
        return "ok"

    def _fmt_percepts(self, p: dict) -> str:
        active = [k.upper() for k, v in p.items() if v]
        return ", ".join(active) if active else "NONE"

    # ── Run full episode ──────────────────────────────────────────────────
    def run(self, max_steps: int = 200, verbose: bool = True, delay: float = 0.0) -> str:
        for step_num in range(1, max_steps + 1):
            if verbose:
                clear_screen()
                print_header()
                print_world(self.world, self)
                print_metrics(self, step_num)
                print_kb_summary(self.kb)
                print_log_tail(self.log, n=18)
                if delay > 0:
                    time.sleep(delay)

            result = self.step()

            if result in ("won", "dead", "stuck"):
                if verbose:
                    clear_screen()
                    print_header()
                    print_world(self.world, self)
                    print_metrics(self, step_num)
                    print_kb_summary(self.kb)
                    print_log_tail(self.log, n=18)
                return result

        return "timeout"


# ─── Display Functions ────────────────────────────────────────────────────────
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print(clr("╔══════════════════════════════════════════════════════════════╗", C.BLUE))
    print(clr("║     WUMPUS WORLD — KNOWLEDGE-BASED PROPOSITIONAL LOGIC AGENT ║", C.BLUE))
    print(clr("╚══════════════════════════════════════════════════════════════╝", C.BLUE))
    print()


def print_world(world: WumpusWorld, agent: LogicAgent):
    """Render the grid. Agent sees only what it has visited / inferred."""
    print(clr("  GRID  (🤖=Agent  💰=Gold  🐛=Wumpus  🕳=Pit  ?=Unknown)", C.WHITE))
    print()

    # Column headers
    header = "     " + "  ".join(clr(f"C{c}", C.GRAY) for c in range(world.cols))
    print(header)
    print()

    for r in range(world.rows - 1, -1, -1):
        row_str = clr(f"R{r} |", C.GRAY) + " "
        for c in range(world.cols):
            pos = (r, c)
            cell = world.grid[pos]
            is_agent = agent.pos == pos
            is_visited = pos in agent.visited
            is_safe = pos in agent.safe_set
            is_danger = pos in agent.danger_set

            if is_agent and not agent.alive:
                sym = clr(" ☠ ", C.BG_RED + C.WHITE)
            elif is_agent:
                sym = clr(" 🤖", C.BG_BLUE)
            elif is_visited:
                if cell.pit:
                    sym = clr(" 🕳 ", C.RED)
                elif cell.wumpus:
                    sym = clr(" 🐛 ", C.YELLOW)
                elif cell.gold and not agent.won:
                    sym = clr(" 💰 ", C.YELLOW)
                else:
                    sym = clr(" ✓  ", C.GREEN)
            elif is_danger:
                sym = clr(" ✗  ", C.RED)
            elif is_safe:
                sym = clr(" ·  ", C.CYAN)
            else:
                sym = clr(" ?  ", C.GRAY)

            row_str += sym + " "
        print(row_str)
    print()

    # Legend
    print(clr("  Legend:", C.WHITE))
    legend = [
        (clr("✓", C.GREEN),  "Visited-safe"),
        (clr("·", C.CYAN),   "Inferred-safe"),
        (clr("?", C.GRAY),   "Unknown"),
        (clr("✗", C.RED),    "Inferred-danger"),
        (clr("🕳", ""),       "Pit (revealed)"),
        (clr("🐛", ""),       "Wumpus (revealed)"),
        (clr("💰", ""),       "Gold"),
    ]
    row = "  "
    for sym, label in legend:
        row += f"{sym} {clr(label, C.GRAY)}  "
    print(row)
    print()


def print_metrics(agent: LogicAgent, step: int):
    w = agent.world
    total = w.rows * w.cols
    visited_pct = len(agent.visited) / total * 100

    print(clr("━" * 64, C.BLUE))
    print(clr("  REAL-TIME METRICS DASHBOARD", C.WHITE + C.BOLD))
    print(clr("━" * 64, C.BLUE))
    items = [
        ("Step #",          clr(step, C.WHITE)),
        ("Moves",           clr(agent.move_count, C.PURPLE)),
        ("Inference Steps", clr(agent.kb.inference_steps, C.CYAN)),
        ("KB Clauses",      clr(agent.kb.clause_count(), C.BLUE)),
        ("Safe Cells",      clr(len(agent.safe_set), C.GREEN)),
        ("Danger Cells",    clr(len(agent.danger_set), C.RED)),
        ("Visited",         clr(f"{len(agent.visited)}/{total} ({visited_pct:.0f}%)", C.YELLOW)),
        ("Agent Status",    clr("ALIVE ✓", C.GREEN) if agent.alive else clr("DEAD ✗", C.RED)),
        ("Position",        clr(agent.pos, C.WHITE)),
    ]
    for i in range(0, len(items), 3):
        row = ""
        for label, val in items[i:i+3]:
            row += f"  {clr(label+':', C.GRAY)} {val:<30}"
        print(row)
    print(clr("━" * 64, C.BLUE))
    print()


def print_kb_summary(kb: KnowledgeBase):
    print(clr("  KNOWLEDGE BASE (last 10 CNF clauses):", C.WHITE))
    clauses = kb.recent_clauses(10)
    for i, cl in enumerate(clauses):
        num = clr(f"  [{len(kb.clauses)-len(clauses)+i+1:3d}]", C.GRAY)
        print(f"{num} ({clr(cl, C.CYAN)})")
    print()


def print_log_tail(log: list, n: int = 15):
    print(clr("  AGENT LOG:", C.WHITE))
    tail = log[-n:]
    for msg, colour in tail:
        print(f"  {colour}{msg}{C.RESET}")
    print()


def print_final_result(result: str, agent: LogicAgent):
    print()
    print(clr("═" * 64, C.BLUE))
    if result == "won":
        print(clr("  🏆  AGENT WON! Gold successfully retrieved!", C.YELLOW + C.BOLD))
    elif result == "dead":
        print(clr("  ☠   AGENT DIED. Better luck next time.", C.RED + C.BOLD))
    elif result == "stuck":
        print(clr("  ⚙   AGENT STUCK. No safe moves available.", C.PURPLE + C.BOLD))
    elif result == "timeout":
        print(clr("  ⏱   TIMEOUT. Max steps reached.", C.YELLOW))

    print(clr("═" * 64, C.BLUE))
    print()
    print(clr("  Final Statistics:", C.WHITE))
    print(f"  {clr('Moves made:',          C.GRAY)} {clr(agent.move_count, C.WHITE)}")
    print(f"  {clr('Inference steps:',     C.GRAY)} {clr(agent.kb.inference_steps, C.CYAN)}")
    print(f"  {clr('KB clauses learned:',  C.GRAY)} {clr(agent.kb.clause_count(), C.BLUE)}")
    print(f"  {clr('Safe cells found:',    C.GRAY)} {clr(len(agent.safe_set), C.GREEN)}")
    print(f"  {clr('Dangers identified:',  C.GRAY)} {clr(len(agent.danger_set), C.RED)}")
    print(f"  {clr('Cells visited:',       C.GRAY)} {clr(len(agent.visited), C.YELLOW)}")
    print()


# ─── Menu ─────────────────────────────────────────────────────────────────────
def get_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            v = int(raw) if raw else default
            if lo <= v <= hi:
                return v
            print(f"  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Invalid input — please enter an integer.")


def get_float(prompt: str, default: float) -> float:
    while True:
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            return float(raw) if raw else default
        except ValueError:
            print("  Invalid input.")


def main_menu():
    clear_screen()
    print_header()
    print(clr("  CONFIGURATION", C.WHITE))
    print(clr("  ─────────────────────────────────────", C.GRAY))
    rows  = get_int("Grid rows  (3–8)", default=4, lo=3, hi=8)
    cols  = get_int("Grid cols  (3–8)", default=4, lo=3, hi=8)
    pits  = get_int("Num pits   (1–6)", default=3, lo=1, hi=6)
    delay = get_float("Step delay seconds (0 = instant)", default=0.3)
    steps = get_int("Max steps  (10–500)", default=100, lo=10, hi=500)

    print()
    print(clr("  Starting episode…", C.GREEN))
    time.sleep(0.5)

    world = WumpusWorld(rows, cols, pits)
    agent = LogicAgent(world)
    result = agent.run(max_steps=steps, verbose=True, delay=delay)
    print_final_result(result, agent)

    print(clr("  Play again? (y/n) [y]: ", C.WHITE), end="")
    again = input().strip().lower()
    if again in ("", "y", "yes"):
        main_menu()
    else:
        print(clr("\n  Thanks for playing Wumpus World!\n", C.CYAN))


# ─── Unit Tests ───────────────────────────────────────────────────────────────
def run_tests():
    """Quick sanity-check of KB resolution logic."""
    print(clr("\n  Running unit tests…\n", C.CYAN))

    kb = KnowledgeBase()

    # Test 1: unit propagation — tell ¬A, ask ¬A
    kb.tell([Literal("A", neg=True)])
    assert kb.ask(Literal("A", neg=True)), "Test 1 failed: should prove ¬A"
    print(clr("  [PASS] Test 1: unit clause ¬A provable", C.GREEN))

    # Test 2: cannot prove A when ¬A is in KB
    assert not kb.ask(Literal("A")), "Test 2 failed: should not prove A"
    print(clr("  [PASS] Test 2: A not provable when ¬A known", C.GREEN))

    # Test 3: simple resolution — (A ∨ B) ∧ ¬A → B
    kb2 = KnowledgeBase()
    kb2.tell([Literal("A"), Literal("B")])   # A ∨ B
    kb2.tell([Literal("A", neg=True)])        # ¬A
    assert kb2.ask(Literal("B")), "Test 3 failed: should resolve B"
    print(clr("  [PASS] Test 3: resolution (A∨B)∧¬A ⊢ B", C.GREEN))

    # Test 4: breeze → pit encoding
    kb3 = KnowledgeBase()
    # No breeze at (0,0) → ¬P_1_0 and ¬P_0_1
    kb3.tell([Literal("P_1_0", neg=True)])
    kb3.tell([Literal("P_0_1", neg=True)])
    assert kb3.ask(Literal("P_1_0", neg=True)), "Test 4a failed"
    assert kb3.ask(Literal("P_0_1", neg=True)), "Test 4b failed"
    print(clr("  [PASS] Test 4: no-breeze → adjacent cells pit-free", C.GREEN))

    # Test 5: tautology check
    assert is_tautology([Literal("X"), Literal("X", neg=True)]), "Test 5 failed"
    assert not is_tautology([Literal("X"), Literal("Y", neg=True)]), "Test 5b failed"
    print(clr("  [PASS] Test 5: tautology detection", C.GREEN))

    print(clr("\n  All tests passed! ✓\n", C.GREEN))


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        main_menu()