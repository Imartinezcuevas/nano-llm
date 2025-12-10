# Contributing guidelines

## Commit convention
This project follows Conventional commits.
```php
<type>(<scope>): <short summary>
<body>
<footer>
```

**Allowed types**
* feat: new feature
* fix: bug fix
* refactor: internal code changes without changing behaviour
* test: add or update test
* docs: documentation changes
* build: changes to dependencies, Docker, CI

## Branching strategy
### Main branches
* main -> production-ready
* dev -> integration/staging

## Pull request template
```shell
## What
(Short description of what changes.)

## Why
(Explain the motivation and value.)

## How
(Technical summary of hot it was implemented.)

## Testing
- [ ] Unit tests added
- [ ] Docker local run successful
- [ ] Lint passes

## Notes
(Edge cases, risks, reviewer notes)
```