import ssik


def test_version_is_present() -> None:
    assert isinstance(ssik.__version__, str)
    assert ssik.__version__
