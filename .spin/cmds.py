import os
import subprocess
import sys

import click
from spin import util
from spin.cmds import meson


def get_git_revision_hash(submodule) -> str:
    return subprocess.check_output(["git", "rev-parse", f"@:{submodule}"]).decode("ascii").strip()


@click.command()
@click.argument("slowtest", default=True)
@click.pass_context
def coverage(ctx, slowtest=True):
    """📊 Generate coverage report"""
    if slowtest:
        pytest_args = (
            "-o",
            "python_functions=test_*",
            "treeple",
            "--cov=treeple",
            "--cov-report=xml",
            "--cov-config=pyproject.toml",
            "-k .",
        )
    else:
        pytest_args = (
            "-o",
            "python_functions=test_*",
            "treeple",
            "--cov=treeple",
            "--cov-report=xml",
            "--cov-config=pyproject.toml",
        )

    # The spin `build` command doesn't know anything about `custom_arg`,
    # so don't send it on.
    del ctx.params["slowtest"]

    ctx.invoke(meson.test, pytest_args=pytest_args)


@click.command()
@click.option("--forcesubmodule", is_flag=False, help="Force submodule pull.")
def setup_submodule(forcesubmodule=False):
    """Build treeple using submodules.

    git submodule set-branch -b submodulev3 treeple/_lib/sklearn

    git submodule update --recursive --remote

    To update submodule wrt latest commits:

        git submodule update --init --recursive --remote
        git add -A
        git commit

    This will update the submodule, which then must be commited so that
    git knows the submodule needs to be at a certain commit hash.
    """
    commit_fpath = "./treeple/_lib/commit.txt"
    submodule = "./treeple/_lib/sklearn_fork"
    commit = ""
    current_hash = ""

    # if the forked folder does not exist, we will need to force update the submodule
    if not os.path.exists("./treeple/_lib/sklearn/") or forcesubmodule:
        # update git submodule
        util.run(["git", "submodule", "update", "--init", "--force"])
    else:
        # update git submodule
        util.run(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--force",
            ]
        )

    # get the commit hash if the commmit file exists
    if os.path.exists(commit_fpath):
        with open(commit_fpath, "r") as f:
            commit = f.read().strip()

    # get revision hash
    current_hash = get_git_revision_hash(submodule)

    print(current_hash)
    print(commit)

    # if the commit file doesn't exist or the commit hash is different, we need
    # to update our sklearn repository
    if current_hash == "" or current_hash != commit:
        util.run(
            [
                "mkdir",
                "-p",
                "./treeple/_lib/sklearn/",
            ],
        )
        util.run(
            [
                "touch",
                commit_fpath,
            ],
        )
        print(commit_fpath)
        with open(commit_fpath, "w") as f:
            f.write(current_hash)

        util.run(
            [
                "rm",
                "-rf",
                "treeple/_lib/sklearn",
            ]
        )

        if os.path.exists("treeple/_lib/sklearn_fork/sklearn") and (commit != current_hash):
            util.run(
                [
                    "cp",
                    "-r",
                    "treeple/_lib/sklearn_fork/sklearn",
                    "treeple/_lib/sklearn",
                ]
            )


@click.command()
@click.argument("meson_args", nargs=-1)
@click.option("-j", "--jobs", help="Number of parallel tasks to launch", type=int)
@click.option("--clean", is_flag=True, help="Clean build directory before build")
@click.option("-v", "--verbose", is_flag=True, help="Print all build output, even installation")
@click.option(
    "--gcov",
    is_flag=True,
    help="Enable C code coverage using `gcov`. Use `spin test --gcov` to generate reports.",
)
@click.option(
    "--forcesubmodule", is_flag=True, help="Force submodule pull.", envvar="FORCE_SUBMODULE"
)
@click.pass_context
def build(
    ctx,
    meson_args,
    jobs=None,
    clean=False,
    verbose=False,
    gcov=False,
    forcesubmodule=False,
):
    """Build treeple using submodules.

        git submodule update --recursive --remote

    To update submodule wrt latest commits:

        git submodule update --init --recursive --remote
        git add -A
        git commit

    This will update the submodule, which then must be commited so that
    git knows the submodule needs to be at a certain commit hash.
    """
    ctx.invoke(setup_submodule, forcesubmodule=forcesubmodule)

    # The spin `build` command doesn't know anything about `custom_arg`,
    # so don't send it on.
    del ctx.params["forcesubmodule"]

    # run build as normal
    # Call the built-in `build` command, passing along
    # all arguments and options.
    ctx.forward(meson.build)


@click.command()
@click.argument("asv_args", nargs=-1)
def asv(asv_args):
    """🏃 Run `asv` to collect benchmarks

    ASV_ARGS are passed through directly to asv, e.g.:

    spin asv -- dev -b TransformSuite

    ./spin asv -- continuous --verbose --split --bench ObliqueRandomForest origin/main constantsv2

    Please see CONTRIBUTING.txt
    """
    site_path = meson._get_site_packages()
    if site_path is None:
        print("No built treeple found; run `spin build` first.")
        sys.exit(1)

    os.environ["ASV_ENV_DIR"] = "/Users/adam2392/miniforge3"
    os.environ["PYTHONPATH"] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(["asv"] + list(asv_args))
