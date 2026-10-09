import pandas as pd

from scripts.fetch_espn_positions import match_to_players, parse_players
from src.features.names import normalize_name
from src.features.positions import add_eligibility, default_eligibility, parse_eligibility


def test_default_eligibility_cases():
    assert default_eligibility("C", None, 3, 12) == {"C"}
    assert default_eligibility("F", "C", 2, 10) == {"PF", "C"}
    assert default_eligibility("G", None, 8, 3) == {"PG"}
    assert default_eligibility("G", None, 2.5, 3) == {"SG"}
    assert default_eligibility("G", None, 4.5, 3) == {"PG", "SG"}
    assert default_eligibility("G", "F", 3, 5) == {"SG", "SF"}
    assert default_eligibility("F", None, 3, 11) == {"PF"}
    assert "PG" in default_eligibility("F", None, 8, 6)          # playmaking forward
    assert default_eligibility(float("nan"), None, 3, 6) == {"SF", "PF"}  # NaN is not a position


def test_espn_file_overrides_heuristic_and_supplies_adp(tmp_path):
    proj = pd.DataFrame({"player_id": [1, 2], "name": ["Nikola Jokić", "Foo Bar"], "position_1": ["C", "G"],
                         "position_2": [None, None], "r_ast": [10.0, 2.0], "r_reb": [12.0, 3.0]})
    f = tmp_path / "positions.csv"
    pd.DataFrame({"player_id": [1], "name": ["Nikola Jokić"], "eligibility": ["C,PF"], "espn_adp": [2.5]}).to_csv(f, index=False)
    out = add_eligibility(proj, f)
    assert out.loc[0, "elig"] == "PF/C" and out.loc[0, "elig_source"] == "espn" and out.loc[0, "espn_adp"] == 2.5
    assert out.loc[1, "elig_source"] == "heuristic" and pd.isna(out.loc[1, "espn_adp"])
    assert out["is_center"].tolist() == [True, False]
    assert parse_eligibility("pg / sg, G, UTIL") == {"PG", "SG"}


def test_parse_espn_payload_and_name_matching():
    payload = {"players": [
        {"player": {"fullName": "Nikola Jokić", "eligibleSlots": [4, 9, 10, 11, 12, 13], "ownership": {"averageDraftPosition": 2.1}}},
        {"player": {"fullName": "Jaren Jackson Jr.", "eligibleSlots": [3, 4, 5, 11], "ownership": {"averageDraftPosition": 31.0}}},
        {"player": {"fullName": "No Slots", "eligibleSlots": [11, 12]}},
        {"player": {"fullName": "Default Only", "defaultPositionId": 1, "eligibleSlots": [11]}},
    ]}
    espn = parse_players(payload)
    assert espn.set_index("espn_name")["eligibility"].to_dict() == {
        "Nikola Jokić": "C", "Jaren Jackson Jr.": "PF,C", "Default Only": "PG"}
    players = pd.DataFrame({"player_id": [10, 20], "name": ["Nikola Jokic", "Jaren Jackson"]})
    m = match_to_players(espn, players)
    assert m["player_id"].tolist() == [10, 20] and m["eligibility"].tolist() == ["C", "PF,C"]
    assert normalize_name("P.J. Washington Jr.") == "pj washington"
