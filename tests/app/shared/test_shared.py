# Placeholder to keep the test directory structure perfectly mirrored with src/
def test_shared_module_exists():
    import src.app.shared
    assert src.app.shared is not None
