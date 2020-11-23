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

To mechanically handle versioning within the package, we use
`setuptools-scm <https://pypi.org/project/setuptools-scm/>`_. This means there's only
one place that versions are stored -- in the git tag. It also offers benefits like the
current version having full version info in it (how many commits since last tag, the
current git hash, etc).

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
4. Update the changelog with a *new version heading* and your fix (link to the issue/PR!)
5. Get a PR review and ensure CI passes.
6. Merge into ``master`` -- don't delete the branch!
7. Create another PR and merge into ``dev`` -- now you can delete
7. Locally, git tag it with the next patch version, and push the tag.

Note that you don't have to merge fixes in this way. You can instead just branch off
``dev``, but then the fix won't be included until the next minor version. This is easier
and useful for non-urgent fixes.

Any other fix/feature should be branched from ``dev``. Every PR that does anything
noteworthy should have an accompanying edit to the changelog. However, you do not have
to update the version in the changelog -- that is left up to the admin(s). To make a
minor release, they should:

1. Make a new branch from ``dev`` called ``release``.
2. No new features should be merged into ``dev`` after that branching occurs.
3. On ``release``, update the CHANGELOG so that the correct version is displayed at the
   top, and add any other overall info to the section.
4. Ensure all tests pass.
5. Merge into ``dev`` -- don't delete!
6. Merge into ``master`` -- now delete.
7. Locally, git tag it with the next minor vesion, and push the tag.

The above also works for MAJOR versions, however getting them *in* to ``dev`` is a little
different, in that they should wait for merging until we're sure that the next version
will be a major version.
