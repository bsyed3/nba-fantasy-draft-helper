"""The data-freshness note: what each refresh records, and how the sidebar words it."""
import datetime as dt

from src.data_ingestion import data_status as D

TODAY = dt.date(2026, 10, 27)
LIVE = {"last_game_date": "2026-10-26", "fetched_at": "2026-10-27T11:48:09", "games": "52"}


def test_record_and_read_round_trip_and_keep_other_entries(tmp_path):
    assert D.read(tmp_path) == {}
    D.record("injuries", out_dir=tmp_path, at=dt.datetime(2026, 10, 27, 11, 37, 44), listings=83, out=61)
    D.record("library", out_dir=tmp_path, at=dt.datetime(2026, 10, 27, 11, 40, 36), year=2026)
    s = D.read(tmp_path)
    assert s["injuries"] == {"at": "2026-10-27T11:37:44", "listings": 83, "out": 61} and s["library"]["year"] == 2026
    D.record("injuries", out_dir=tmp_path, at=dt.datetime(2026, 10, 28, 9, 0), listings=70, out=50)    # a refresh replaces its own entry only
    s = D.read(tmp_path)
    assert s["injuries"]["out"] == 50 and "library" in s and not list(tmp_path.glob("*.tmp"))


def test_a_corrupt_status_file_reads_as_empty(tmp_path):
    (tmp_path / D.FILE).write_text("{not json", encoding="utf-8")
    assert D.read(tmp_path) == {}


def test_sidebar_lines_say_how_far_the_game_stats_run_and_when_the_rest_was_refreshed():
    status = {"injuries": {"at": "2026-10-27T11:37:44", "out": 61}, "projections": {"at": "2026-10-27T11:38:50"}, "library": {"at": "2026-10-27T11:40:36"}}
    lines = dict(D.describe(status, LIVE, TODAY))
    assert lines["Game stats"] == "through Oct 26 (yesterday), 52 games · pulled Oct 27, 11:48 AM"
    assert lines["Injuries (ESPN)"] == "as of Oct 27, 11:37 AM (today) · 61 players out"
    assert lines["Projections"].startswith("built Oct 27, 11:40 AM") and "game stats are not used" in lines["Projections"]
    assert "Warning" not in lines


def test_missing_pieces_say_how_to_get_them_and_a_stale_library_is_flagged():
    lines = dict(D.describe({}, None, TODAY))
    assert "fetch_live_stats" in lines["Game stats"] and "fetch_espn_injuries" in lines["Injuries (ESPN)"] and lines["Projections"] == "no build recorded"
    stale = {"projections": {"at": "2026-10-27T12:00:00"}, "library": {"at": "2026-10-27T11:40:00"}}
    assert "build_library" in dict(D.describe(stale, LIVE, TODAY))["Warning"]
    older = dict(D.describe({}, {**LIVE, "last_game_date": "2026-10-22"}, TODAY))
    assert "5 days ago" in older["Game stats"]
