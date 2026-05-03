import sys
import pytest
import pyrodeen.cli as cli


def test_parse():
    parsed_args_t = cli.parse(["-t"])
    parsed_args = cli.parse([])
    assert(parsed_args_t.test_cli)
    assert(not parsed_args.test_cli)

#Instead of doing two asserts, we can parametrize the test to check both cases
#(Will be executed twice, once for each set of parameters)
@pytest.mark.parametrize(
    "argv, expected",
    [
        (["pyrodeen", "-t"], cli.cli_test_message),
        (["pyrodeen"], ""),
    ],
)
def test_main(argv, expected, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    captured = capsys.readouterr()
    assert(captured.out.strip() == expected)