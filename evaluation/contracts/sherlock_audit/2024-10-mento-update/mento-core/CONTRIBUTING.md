# Contributing to Mento Core

The following is a set of guidelines for contributing to the Mento protocol. Reading and following the guidelines will help make the contribution process easy and allow us to maintain a consistent standard of code & quality.

## Quick Links

[How To Contribute](#how-to-contribute)

- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features/Enhancements](#suggesting-features)
- [Code Contributions](#code-contributions)

[Styleguides](#styleguides)

- [Git Commit Messages](#git-commit-messages)

## How To Contribute

There are many ways you can contribute to the development of Mento. Contributions can be made to this repo via issues and pull requests. However, before contributing, you should have a good understanding of how Mento is intended to work and the different components that make up the core protocol. You can learn more about Mento by taking some time to read the following:

- [Protocol Documentation](https://docs.mento.org/mento-protocol/core/overview)
- [Stability Whitepaper](https://celo.org/papers/stability)

### Reporting Bugs

Before making a bug report, check the list of open issues, as you may find that a report already exists for the same bug. You can report a bug by creating an issue and selecting the bug report template. When filling out the template, please include as much detail as possible.

### Suggesting Features

You can contribute to the development of Mento by suggesting new features to be built or enhancements to existing functionality. Before proposing a feature, check that an issue does not already exist for the feature or enhancement. To suggest a feature or enhancement, create a new issue and select the feature request template. Once your issue has been created, it's best to start a discussion in the Mento [discord server](http://chat.mento.org) referencing the issue.

### Code Contributions

Code contributions can be made by creating a pull request that addresses an open issue labelled with **good first issue** or **help wanted**. PR's that address styling issues or add unit/integration tests are always welcome. For changes that address core functionality or would require breaking changes, it's best to open an Issue to discuss your proposal first.

## Naming conventions

Our naming conventions have the goal of making our code cleaner and our lives easier.

### Git branches

We follow the [Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) and allow for the following branch names:

- Production branch: main
- Develop branch: develop
- Feature prefix: feature/\*
- Release prefix: release/\*
- Hotfix prefix: hotfix/\*

### Git Commit Messages

Below are some guidelines for git commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Follow [Semantic Commit Messages](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716)
- Consider ending commit description using relevant emoji (e.g. test: verify token address is valid 🕵️)

Semantic Commit Messages allow the following commit message style:

`<type>(<scope>): <subject>`
where
`<scope>` is optional and `<type>` can be any of

- feat: (new feature for the user, not a new feature for build script)
- fix: (bug fix for the user, not a fix to a build script)
- docs: (changes to the documentation)
- style: (formatting, missing semi colons, etc; no production code change)
- refactor: (refactoring production code, eg. renaming a variable)
- test: (adding missing tests, refactoring tests; no production code change)
- chore: (updating grunt tasks etc; no production code change)

### Git Pull Requests

Pull requests should be named as a mix from branch name and commit message style.

`<type>(<scope>): <branch-name>`, for example `docs(contributing): feature/naming-conventions` where `feature/naming-conventions`is the branch name.
