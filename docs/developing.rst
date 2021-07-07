Developing hmf
==============

If you're interested in developing hmf -- welcome! You should first read
the guide to contributing. This page is about more technical details of how
hmf is developed, and its philosophy.


Branching and Releasing
-----------------------
The aim is to make hmf's releases as useful, comprehendible, and automatic
as possible. This section lays out explicitly how this works (mostly for the benefit of
the admin(s)).

Versioning
~~~~~~~~~~
The first thing to mention is that we use strict `semantic versioning <https://semver.org>`_
(since v2.0). Thus the versions are ``MAJOR.MINOR.PATCH``, with ``MAJOR`` including
API-breaking changes, ``MINOR`` including new features, and ``PATCH`` fixing bugs or
documentation etc. If you depend on hmf, you can set your dependency as
``hmf >= X.Y < X+1`` and not worry that we'll break your code with an update.

To mechanically handle versioning within the package, we use two methods that we make
to work together automatically. The "true" version of the package is set with
`setuptools-scm <https://pypi.org/project/setuptools-scm/>`_. This stores the version
in the git tag. There are many benefits to this -- one is that the version is unique
for every single change in the code, with commits on top of a release changing the
version. This means that versions accessed via ``hmf.__version__`` are unique and track
the exact code in the package (useful for reproducing results). To get the current
version from command line, simply do ``python setup.py --version`` in the top-level
directory.

To actually bump the version, we use ``bump2version``. The reason for this is that the
CHANGELOG requires manual intervention -- we need to change the "dev-version" section
at the top of the file to the current version. Since this has to be manual, it requires
a specific commit to make it happen, which thus requires a PR (since commits can't be
pushed to master). To get all this to happen as smoothly as possible, we have a little
bash script ``bump`` that should be used to bump the version, which wraps ``bump2version``.
What it does is:

1. Runs ``bump2version`` and updates the ``major``, ``minor`` or ``patch`` part (passed like
   ``./bump minor``) in the VERSION file.
2. Updates the changelog with the new version heading (with the date),
   and adds a new ``dev-version`` heading above that.
3. Makes a commit with the changes.

.. note:: Using the ``bump`` script is currently necessary, but future versions of
   ``bump2version`` may be able to do this automatically, see
   https://github.com/c4urself/bump2version/issues/133.

The VERSION file might seem a bit redundant, and it is NOT recognized as the "official"
version (that is given by the git tag). Notice we didn't make a git tag in the above
script. That's because the tag should be made directly on the merge commit into master.
We do this using a Github Action (``tag-release.yaml``) which runs on every push to master,
reads the VERSION file, and makes a tag based on that version.


Branching
~~~~~~~~~
For branching, we use a very similar model to `git-flow <https://nvie.com/posts/a-successful-git-branching-model/>`_.
That is, we have a ``dev`` branch which acts as the current truth against which to develop,
and ``master`` essentially as a deployment branch.
I.e., the ``dev`` branch is where all features are merged (and some
non-urgent bugfixes). ``master`` is always production-ready, and corresponds
to a particular version on PyPI. Features should be branched from ``dev``,
and merged back to ``dev``. Hotfixes can be branched directly from ``master``,
and merged back there directly, *as well as* back into ``dev``.
*Breaking changes* must only be merged to ``dev`` when it has been decided that the next
version will be a major version. We do not do any long-term support of releases
(so can't make hotfixes to ``v2.x`` when the latest version is ``2.(x+1)``, or make a
new minor version in 2.x when the latest version is 3.x). We have set the default
branch to ``dev`` so that by default, branches are merged there. This is deemed best
for other developers (not maintainers/admins) to get involved, so the default thing is
usually right.

.. note:: Why not a more simple workflow like Github flow? The simple answer is it just
          doesn't really make sense for a library with semantic versioning. You get into
          trouble straight away if you want to merge a feature but don't want to update
          the version number yet (you want to merge multiple features into a nice release).
          In practice, this happens quite a lot.

.. note:: OK then, why not just use ``master`` to accrue features and fixes until such
          time we're ready to release? The problem here is that if you've merged a few
          features into master, but then realize a patch fix is required, there's no
          easy way to release that patch without releasing all the merged features, thus
          updating the minor version of the code (which may not be desirable). You could
          then just keep all features in their own branches until you're ready to release,
          but this is super annoying, and doesn't give you the chance to see how they
          interact.


Releases
~~~~~~~~
To make a **patch** release, follow these steps:

1. Branch off of ``master``.
2. Write the fix.
3. Write a test that would have broken without the fix.
4. Update the changelog with your changes, under the ``**Bugfixes**`` heading.
5. Commit, push, and create a PR.
6. Locally, run ``./bump patch``.
7. Push.
8. Get a PR review and ensure CI passes.
9. Merge the PR

Note that in the background, Github Actions *should* take care of then tagging master
with the new version, deploying that to PyPI, creating a new PR from master back into
dev, and accepting that PR. If it fails for one of these steps, they can all be done
manually.

Note that you don't have to merge fixes in this way. You can instead just branch off
``dev``, but then the fix won't be included until the next ``minor`` version.
This is easier (the admins do the adminy work) and useful for non-urgent fixes.

Any other fix/feature should be branched from ``dev``. Every PR that does anything
noteworthy should have an accompanying edit to the changelog. However, you do not have
to update the version in the changelog -- that is left up to the admin(s). To make a
minor release, they should:

1. Locally, ``git checkout release``
2. ``git merge dev``
3. No new features should be merged into ``dev`` after that branching occurs.
4. Run ``./bump minor``
5. Make sure everything looks right.
6. ``git push``
7. Ensure all tests pass and get a CI review.
8. Merge into ``master``

The above also works for ``MAJOR`` versions, however getting them *in* to ``dev`` is a little
different, in that they should wait for merging until we're sure that the next version
will be a major version.
