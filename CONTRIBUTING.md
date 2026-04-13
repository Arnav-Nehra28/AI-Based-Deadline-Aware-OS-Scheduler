# Contributing

Thanks for taking a look at the project.

## Development Guidelines

- Keep source changes focused and small.
- Prefer updating existing patterns rather than introducing a second style.
- Do not commit generated artifacts, trained models, local datasets, or report outputs.
- Add or update tests when behavior changes.

## Recommended Workflow

1. Create a branch for your change.
2. Make the code update.
3. Run the relevant tests.
4. Check `git status` and confirm only source files are staged.
5. Open a pull request with a short explanation of the change.

## Before Opening a PR

Please verify:

- the code runs in a clean virtual environment
- ignored artifacts are not staged
- tests still pass
- documentation is updated if behavior or usage changed
