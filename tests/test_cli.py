from autofdtd.cli import main


def test_cli_json_output(capsys) -> None:
    code = main(["--format", "json"])
    captured = capsys.readouterr()
    assert code == 0
    assert '"feature": "Simulation"' in captured.out


def test_cli_table_output(capsys) -> None:
    code = main(["--format", "table"])
    captured = capsys.readouterr()
    assert code == 0
    assert "AutoFDTD Phase 1 feature matrix:" in captured.out
