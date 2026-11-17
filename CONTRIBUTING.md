# Contributing to blockyOS

Thanks for helping. blockyOS is a measurement tool, so the bar for changes that affect numbers is higher than usual.

## Before you start

- Read [AGENTS.md](AGENTS.md): repository layout, the checks to run, and the project rules. They apply to people too.
- Read [docs/PROTOCOL.md](docs/PROTOCOL.md) if your change touches how anything is measured.
- For anything bigger than a fix, open an issue first so the approach can be agreed before you write the code.

## Development setup

- Backend: Python 3.12 or newer. `pip install -r backend/requirements-dev.txt`, then `cd backend && python -m pytest -q`.
  The tests need no GPU, models or network: every path points into a temporary folder.
- Frontend: Node.js 22. `cd frontend && npm ci && npm test && npm run build`.
- Lint: `ruff check backend client` (configuration in `pyproject.toml`).

## Rules that protect results

1. **Measurement changes are protocol changes.** If you change how a metric is measured, bump `PROTOCOL` in
   `backend/app/profile.py`, update docs/PROTOCOL.md and add an entry to CHANGELOG.md. Results from different protocol
   versions must never be compared silently.
2. **Report uncertainty.** New metrics need a spread or interval and, where it applies, a test. The UI must not state
   conclusions the data doesn't support.
3. **Settings that change a measurement** go into each run's provenance and into the comparison's mismatch checks.
4. **Local only by default.** Nothing new may send data off the device or listen on the network without an explicit
   user setting.
5. **No destructive tools.** The MCP server and background jobs never delete user data.
6. **Tests come with the change.** Bug fixes get a test that fails without the fix.

## Pull requests

- Keep each pull request to one topic and describe what changed in behavior, not only in code.
- Say how you tested it, including on what hardware if it touches measurement or GPU code.
- CI (backend tests on Python 3.12 and 3.13, lint, frontend tests and build) must pass.

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE), as stated in
its section 5.
