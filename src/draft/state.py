"""
League settings and the live draft state.

The draft state is nothing but an ordered pick log (pick_no, team_slot,
player_id). Who is on the clock, every roster, and the remaining pool are
derived from it, so "undo" is just dropping the last pick and the log is
all that needs persisting (see store.py).

ESPN snake draft: team slots 1..n pick 1..n in round 1, n..1 in round 2,
and so on.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeagueSettings:
    name: str = "ESPN 10-team"
    n_teams: int = 10
    roster_size: int = 13        # drafted players: 10 active + 3 bench (the IR slot is not drafted -- see docs)
    active_slots: int = 10       # PG SG G SF PF F C + 3 UTIL, set daily
    my_slot: int = 1             # 1-based first-round position
    max_centers: int = 3         # league rule: at most this many centers on a roster
    season: str = "2026-27"
    reg_season_weeks: int = 18
    playoff_weeks: int = 3

    @property
    def total_picks(self) -> int:
        return self.n_teams * self.roster_size


def team_for_pick(pick_no: int, n_teams: int) -> int:
    """1-based team slot that owns 1-based overall pick `pick_no` in a snake draft."""
    rnd, pos = divmod(pick_no - 1, n_teams)
    return pos + 1 if rnd % 2 == 0 else n_teams - pos


def snake_order(n_teams: int, rounds: int) -> list[int]:
    return [team_for_pick(p, n_teams) for p in range(1, n_teams * rounds + 1)]


@dataclass
class DraftState:
    settings: LeagueSettings
    picks: list[tuple[int, int, int]] = field(default_factory=list)  # (pick_no, team_slot, player_id)
    excluded: set[int] = field(default_factory=set)                   # flagged unavailable (injury etc.)
    centers: set[int] = field(default_factory=set)                    # player_ids that count toward the center cap

    # --- derived -------------------------------------------------------
    @property
    def next_pick_no(self) -> int:
        return len(self.picks) + 1

    @property
    def done(self) -> bool:
        return len(self.picks) >= self.settings.total_picks

    @property
    def team_on_clock(self) -> int | None:
        return None if self.done else team_for_pick(self.next_pick_no, self.settings.n_teams)

    @property
    def my_turn(self) -> bool:
        return self.team_on_clock == self.settings.my_slot

    @property
    def drafted(self) -> set[int]:
        return {p for _, _, p in self.picks}

    def roster(self, slot: int) -> list[int]:
        return [p for _, t, p in self.picks if t == slot]

    def rosters(self) -> dict[int, list[int]]:
        out = {t: [] for t in range(1, self.settings.n_teams + 1)}
        for _, t, p in self.picks:
            out[t].append(p)
        return out

    def my_pick_numbers(self) -> list[int]:
        s = self.settings
        return [p for p in range(1, s.total_picks + 1) if team_for_pick(p, s.n_teams) == s.my_slot]

    def my_next_pick_no(self) -> int | None:
        return next((p for p in self.my_pick_numbers() if p >= self.next_pick_no), None)

    def picks_until_my_turn(self) -> int | None:
        nxt = self.my_next_pick_no()
        return None if nxt is None else nxt - self.next_pick_no

    def n_centers(self, slot: int) -> int:
        return sum(1 for p in self.roster(slot) if p in self.centers)

    def can_draft(self, player_id: int, slot: int | None = None) -> bool:
        """False if already drafted or it would exceed the team's center cap."""
        slot = slot or self.team_on_clock
        if player_id in self.drafted or slot is None:
            return False
        return not (player_id in self.centers and self.n_centers(slot) >= self.settings.max_centers)

    # --- mutation ------------------------------------------------------
    def make_pick(self, player_id: int) -> tuple[int, int, int]:
        if self.done:
            raise ValueError("draft is complete")
        if player_id in self.drafted:
            raise ValueError(f"player {player_id} was already drafted")
        if not self.can_draft(player_id):
            raise ValueError(f"team {self.team_on_clock} already has {self.settings.max_centers} centers")
        pick = (self.next_pick_no, self.team_on_clock, int(player_id))
        self.picks.append(pick)
        self.excluded.discard(int(player_id))
        return pick

    def undo(self) -> tuple[int, int, int] | None:
        return self.picks.pop() if self.picks else None
