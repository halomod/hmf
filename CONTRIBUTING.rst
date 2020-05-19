Contributing to hmf
===================
Thank you for considering contributing to hmf!

hmf is an open source project, and will only get better as the community of contributors
grows.
There are many ways to contribute, including writing tutorials,
improving the documentation, submitting bug reports and feature requests or writing code
which can be incorporated into hmf itself. Following the guidelines and patterns
suggested below helps us maintain a high quality code base and review your
contributions faster.

How to report a bug
-------------------
First check the issues to see if your bug already exists. Feel free to comment on the
existing issue to provide more context or just to note that it is affecting you as well.
If your bug is not in the issue list, make a new issue.

When making an issue, try to provide as much context as possible including:

1. What version of python and hmf are you using?
2. What operating system are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?
6. Any code to reproduce the bug (as minimal an example as possible)

If you're really inspired, you can make a pull request adding a test that fails because
of the bug. This is likely to lead to the bug being fixed more quickly.

How to suggest a feature or enhancement
---------------------------------------
First check the issues to see if your feature request already exists. Feel free to
comment on the existing issue to provide more context or just to note that you would
like to see the feature implemented as well. If your feature request is not in the issue
list, make a new issue.

When making a feature request, try to provide as much context as possible.
Feel free to include suggestions for implementations.

Guidelines for contributing to the code
---------------------------------------
* Create issues for any major changes and enhancements that you wish to make. Discuss
  things transparently and get community feedback.
* Keep pull requests as small as possible. Ideally each pull request should implement
  ONE feature or bugfix. If you want to add or fix more than one thing, submit more than
  one pull request.
* Do not commit changes to files that are irrelevant to your feature or bugfix.
* Be aware that the pull request review process is not immediate, and is generally
  proportional to the size of the pull request.

Your First Contribution
~~~~~~~~~~~~~~~~~~~~~~~
Contributing for the first time can seem daunting, but we value contributions from our
user community and we will do our best to help you through the process. Here’s some
advice to help make your work on hmf more useful and rewarding.

* Use issue labels to guide you
  - Unsure where to begin contributing to hmf? You can start by looking through issues
    labeled `good first issue` and `help wanted` issues.
* Pick a subject area that you care about, that you are familiar with, or that you want
  to learn about
  - There are many aspects to hmf, from cosmography, through cosmological initial
    conditions, through to filter functions and fitting functions. Choose the one
    you're most interested in!
* Start small
  - It’s easier to get feedback on a little issue or pull request than on a big one.
* If you’re going to take on a big change, make sure that your idea has support first
  - This means getting someone else to confirm that a bug is real before you fix the
    issue, and ensuring that there’s consensus on a proposed feature before you work to
    implement it. Use the issue log to start conversations about major changes and
    enhancements.
* Be bold! Leave feedback!
  - Sometimes it can be scary to make new issues or comment on existing issues or pull
    requests, but contributions from the wider community are what ensure that hmf serves
    the whole community as well as possible.
* Be rigorous
  - Our requirements on code style, testing and documentation are important. If you have
    questions about them or difficulty meeting them, please ask for help, we will do our
    best to support you. Your contributions will be reviewed and integrated much more
    quickly if your pull request meets the requirements.

If you are new to the GitHub or the pull request process you can start by taking a look
at these tutorials: http://makeapullrequest.com/ and http://www.firsttimersonly.com/.
If you have more questions, feel free to ask for help, everyone is a beginner at first
and all of us are still learning!

Getting started
---------------
1. Create your own fork or branch of the code.
2. Follow the [Developer Installation](INSTALLATION.rst) instructions to ensure that you
   have all the required packages for testing your changes.
3. Run ``pre-commit install`` to enable code-quality checks.
4. Make the changes in your fork or branch.
5. If you like the change and think the project could use it:
  - If you're fixing a bug, include a new test that breaks as a result of the bug (if possible).
  - Ensure that all your new code is covered by tests and that the existing tests pass.
    Tests can be run by running `pytest` in the top level ``hmf`` directory.
  - Ensure that you fully document any new features via docstrings, and potentially
    as a new tutorial in the `docs/` directory.
6. Make a Pull Request from your fork/branch.

Code review process
-------------------
The core team looks at pull requests on a regular basis and tries to provide feedback as
quickly as possible. Larger pull requests generally require more time for review.

Release Cycle and Versioning
----------------------------
In the past ``hmf`` has used ``git-flow`` as a release management workflow. This is,
however, quite complicated to enable properly through GitHub, and we have fallen back
on the simpler `Github-Flow<https://guides.github.com/introduction/flow/>`_.

In this workflow, ``master`` is protected and commits may *not* be pushed to it directly,
but must first undergo testing and review via a Pull Request.

From v3.1.0, ``hmf`` will be using strict semantic versioning, such that increases in
the **major** version have potential API breaking changes, **minor** versions introduce
new features, and **patch** versions fix bugs and other non-breaking internal changes.

The versions are controlled by git tags, via the ``setuptools_scm`` package. Most
bugfixes and new features will be automatically deployed to ``PyPI`` (so installable
via ``pip install hmf``) immediately upon accepting the Pull Request, as well as the
relevant version being updated. This is done via *tags* in the Pull Request to inform
the deployment tools which version should be increased.

This process automates as much as possible while also not leaving feature branches to
gather dust waiting to be deployed. However, one potential problem then remains: what
if a new feature is added that is backwards incompatible (therefore requiring a new
major release), but a new release has just been deployed? Since we will not support
fixing previous releases with bugfixes, and we don't want users to have to update their
scripts too often, we'd like to be able to withhold that from going into master. In this
case, a special branch can be made that will act as a conglomeration point for all
additions to the new version, to enable parallel working on the current and future
branch.
