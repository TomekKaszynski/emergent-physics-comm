# Publishing WMCP to PyPI

## TestPyPI (Pre-Publication)

### Prerequisites
1. Create account at https://test.pypi.org/account/register/
2. Generate API token at https://test.pypi.org/manage/account/token/
3. Save token in `~/.pypirc`:
```ini
[testpypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
```

### Upload
```bash
pip install twine
python -m build
twine upload --repository testpypi dist/*
```

### Verify
```bash
python -m venv /tmp/wmcp-pypi-test
source /tmp/wmcp-pypi-test/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ wmcp
python -m wmcp.cli info
pytest wmcp/tests/ -v
deactivate
```

## Real PyPI (Production)

Same process, replace `testpypi` with `pypi` and use https://pypi.org tokens.

```bash
twine upload dist/*
```

Then anyone can:
```bash
pip install wmcp
wmcp info
```

## Status

- [x] Package builds successfully (`python -m build`)
- [x] Clean venv install works (Phase 131)
- [x] CLI works in clean venv
- [x] 23/23 tests pass in clean venv
- [ ] TestPyPI account created
- [ ] TestPyPI upload verified
- [ ] Production PyPI published
