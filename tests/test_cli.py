import pytest
from conftest import FakeFaces, FakeImmich, photo

from if_curator import cli

PEOPLE = [{"id": "p1", "name": "Ann Lee"}, {"id": "p2", "name": "Anna"}, {"id": "p3", "name": "Bo"}]


@pytest.mark.parametrize("query, expected", [("3", "p3"), ("anna", "p2"), ("lee", "p1"), (" bo ", "p3")])
def test_find_person(query, expected):
    assert cli.find_person(PEOPLE, query)["id"] == expected


@pytest.mark.parametrize("query, message", [("an", "several"), ("zed", "No one"), ("9", "No one")])
def test_find_person_explains_misses(query, message):
    with pytest.raises(LookupError, match=message):
        cli.find_person(PEOPLE, query)


def test_non_interactive_run(monkeypatch, tmp_path, capsys):
    immich = FakeImmich([photo(i) for i in range(12)])
    immich.people = lambda: PEOPLE
    monkeypatch.setenv("IMMICH_URL", "http://immich")
    monkeypatch.setenv("API_KEY", "key")
    monkeypatch.setattr(cli, "Immich", lambda url, key: immich)
    monkeypatch.setattr(cli, "load_model", lambda object_class, settings: (FakeFaces(), cli_analyze()))
    with pytest.raises(SystemExit) as exit:
        cli.main(["Ann Lee", "-n", "4", "--yes"])
    assert exit.value.code == 0
    output = capsys.readouterr().out
    assert "Ann Lee" in output and "Saved to" in output
    (run,) = (tmp_path / "frigate_train").iterdir()
    assert len(list((run / "Ann Lee").iterdir())) >= 1


def cli_analyze():
    from if_curator.faces import analyze_faces

    return analyze_faces


def test_unknown_name_fails_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("IMMICH_URL", "http://immich")
    monkeypatch.setenv("API_KEY", "key")
    immich = FakeImmich([])
    immich.people = lambda: PEOPLE
    monkeypatch.setattr(cli, "Immich", lambda url, key: immich)
    with pytest.raises(SystemExit) as exit:
        cli.main(["Zed"])
    assert exit.value.code == 1 and "No one in Immich is called" in capsys.readouterr().out
