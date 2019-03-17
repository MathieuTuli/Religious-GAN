from imrsv import production


def test_hello(capsys):
    assert production.hello() is None
    out, err = capsys.readouterr()
    assert out == 'World!\n'
    assert not err

    assert production.hello('Say what again!') is None
    out, err = capsys.readouterr()
    assert out == 'Say what again!\n'
    assert not err
