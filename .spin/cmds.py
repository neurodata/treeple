import contextlib
import errno
import os
import shutil
import subprocess
import sys
import warnings
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from sysconfig import get_path

import click
from click import Option
from doit.api import run_tasks
from doit.cmd_base import ModuleTaskLoader
from doit.exceptions import TaskError
from doit.reporter import ZeroReporter
from pydevtool.cli import CliGroup, Task, UnifiedContext
from rich.console import Console
from rich.panel import Panel
from spin import util
from spin.cmds import meson

PROJECT_MODULE = "sktree"


@dataclass
class Dirs:
    """
    root:
        Directory where scr, build config and tools are located
        (and this file)
    build:
        Directory where build output files (i.e. *.o) are saved
    install:
        Directory where .so from build and .py from src are put together.
    site:
        Directory where the built SciPy version was installed.
        This is a custom prefix, followed by a relative path matching
        the one the system would use for the site-packages of the active
        Python interpreter.
    """

    # all paths are absolute
    root: Path
    build: Path
    installed: Path
    site: Path  # <install>/lib/python<version>/site-packages

    def __init__(self, args=None):
        """:params args: object like Context(build_dir, install_prefix)"""
        self.root = Path(__file__).parent.absolute()
        if not args:
            return

        self.build = Path(args.build_dir).resolve()
        if args.install_prefix:
            self.installed = Path(args.install_prefix).resolve()
        else:
            self.installed = self.build.parent / (self.build.stem + "-install")

        if sys.platform == "win32" and sys.version_info < (3, 10):
            # Work around a pathlib bug; these must be absolute paths
            self.build = Path(os.path.abspath(self.build))
            self.installed = Path(os.path.abspath(self.installed))

        # relative path for site-package with py version
        # i.e. 'lib/python3.10/site-packages'
        self.site = self.get_site_packages()

    def add_sys_path(self):
        """Add site dir to sys.path / PYTHONPATH"""
        site_dir = str(self.site)
        sys.path.insert(0, site_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join((site_dir, os.environ.get("PYTHONPATH", "")))

    def get_site_packages(self):
        """
        Depending on whether we have debian python or not,
        return dist_packages path or site_packages path.
        """
        if sys.version_info >= (3, 12):
            plat_path = Path(get_path("platlib"))
        else:
            # distutils is required to infer meson install path
            # for python < 3.12 in debian patched python
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                from distutils import dist
                from distutils.command.install import INSTALL_SCHEMES
            if "deb_system" in INSTALL_SCHEMES:
                # debian patched python in use
                install_cmd = dist.Distribution().get_command_obj("install")
                install_cmd.select_scheme("deb_system")
                install_cmd.finalize_options()
                plat_path = Path(install_cmd.install_platlib)
            else:
                plat_path = Path(get_path("platlib"))
        return self.installed / plat_path.relative_to(sys.exec_prefix)


def get_git_revision_hash(submodule) -> str:
    return subprocess.check_output(["git", "rev-parse", f"@:{submodule}"]).decode("ascii").strip()


def get_test_runner(project_module):
    """
    get Test Runner from locally installed/built project
    """
    __import__(project_module)
    # scipy._lib._testutils:PytestTester
    test = sys.modules[project_module].test
    version = sys.modules[project_module].__version__
    mod_path = sys.modules[project_module].__file__
    mod_path = os.path.abspath(os.path.join(os.path.dirname(mod_path)))
    return test, version, mod_path


@contextlib.contextmanager
def working_dir(new_dir):
    current_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(current_dir)


class ErrorOnlyReporter(ZeroReporter):
    desc = """Report errors only"""

    def runtime_error(self, msg):
        console = Console()
        console.print("[red bold] msg")

    def add_failure(self, task, fail_info):
        console = Console()
        if isinstance(fail_info, TaskError):
            console.print(f"[red]Task Error - {task.name}" f" => {fail_info.message}")
        if fail_info.traceback:
            console.print(
                Panel(
                    "".join(fail_info.traceback),
                    title=f"{task.name}",
                    subtitle=fail_info.message,
                    border_style="red",
                )
            )


CONTEXT = UnifiedContext(
    {
        "build_dir": Option(
            ["--build-dir"],
            metavar="BUILD_DIR",
            default="build",
            show_default=True,
            help=":wrench: Relative path to the build directory.",
        ),
        "no_build": Option(
            ["--no-build", "-n"],
            default=False,
            is_flag=True,
            help=(
                ":wrench: Do not build the project"
                " (note event python only modification require build)."
            ),
        ),
        "install_prefix": Option(
            ["--install-prefix"],
            default=None,
            metavar="INSTALL_DIR",
            help=(
                ":wrench: Relative path to the install directory."
                " Default is <build-dir>-install."
            ),
        ),
    }
)


def run_doit_task(tasks):
    """
    :param tasks: (dict) task_name -> {options}
    """
    loader = ModuleTaskLoader(globals())
    doit_config = {
        "verbosity": 2,
        "reporter": ErrorOnlyReporter,
    }
    return run_tasks(loader, tasks, extra_config={"GLOBAL": doit_config})


class CLI(CliGroup):
    context = CONTEXT
    run_doit_task = run_doit_task


@click.group(cls=CLI)
@click.pass_context
def cli(ctx, **kwargs):
    """Developer Tool for SciPy

    \bCommands that require a built/installed instance are marked with :wrench:.


    \b**python dev.py --build-dir my-build test -s stats**

    """  # noqa: E501
    CLI.update_context(ctx, kwargs)


@click.command()
@click.option("--build-dir", default="build", help="Build directory; default is `$PWD/build`")
@click.option("--clean", is_flag=True, help="Clean previously built docs before building")
@click.option("--noplot", is_flag=True, help="Build docs without plots")
@click.pass_context
def docs(ctx, build_dir, clean=False, noplot=False):
    """📖 Build documentation"""
    if clean:
        doc_dir = "./docs/_build"
        if os.path.isdir(doc_dir):
            print(f"Removing `{doc_dir}`")
            shutil.rmtree(doc_dir)

    site_path = meson._get_site_packages()
    if site_path is None:
        print("No built scikit-tree found; run `./spin build` first.")
        sys.exit(1)

    util.run(["pip", "install", "-q", "-r", "doc_requirements.txt"])

    ctx.invoke(meson.docs)
    # os.environ["SPHINXOPTS"] = "-W"
    # os.environ["PYTHONPATH"] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    # if noplot:
    #     util.run(["make", "-C", "docs", "clean", "html-noplot"], replace=True)
    # else:
    #     util.run(["make", "-C", "docs", "clean", "html"], replace=True)


@click.command()
@click.pass_context
def coverage(ctx):
    """📊 Generate coverage report"""
    pytest_args = ("-o", "python_functions=test_*", "sktree", "--cov=sktree", "--cov-report=xml")
    ctx.invoke(meson.test, pytest_args=pytest_args)


@click.command()
@click.option("--forcesubmodule", is_flag=False, help="Force submodule pull.")
def setup_submodule(forcesubmodule=False):
    """Build scikit-tree using submodules.

    git submodule set-branch -b submodulev2 sktree/_lib/sklearn

    git submodule update --recursive --remote

    To update submodule wrt latest commits:

        git submodule update --init --recursive --remote
        git add -A
        git commit

    This will update the submodule, which then must be commited so that
    git knows the submodule needs to be at a certain commit hash.
    """
    commit_fpath = "./sktree/_lib/sklearn_fork/commit.txt"
    submodule = "./sktree/_lib/sklearn_fork"
    commit = ""
    current_hash = ""

    # if the forked folder does not exist, we will need to force update the submodule
    if not os.path.exists("./sktree/_lib/sklearn/") or forcesubmodule:
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
                "./sktree/_lib/sklearn/",
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
                "sktree/_lib/sklearn",
            ]
        )

        if os.path.exists("sktree/_lib/sklearn_fork/sklearn") and (commit != current_hash):
            util.run(
                [
                    "cp",
                    "-r",
                    "sktree/_lib/sklearn_fork/sklearn",
                    "sktree/_lib/sklearn",
                ]
            )


@click.command()
@click.option("-j", "--jobs", help="Number of parallel tasks to launch", type=int)
@click.option("--clean", is_flag=True, help="Clean build directory before build")
@click.option(
    "--forcesubmodule", is_flag=True, help="Force submodule pull.", envvar="FORCE_SUBMODULE"
)
@click.option("-v", "--verbose", is_flag=True, help="Print all build output, even installation")
@click.argument("meson_args", nargs=-1)
@click.pass_context
def build(ctx, meson_args, jobs=None, clean=False, forcesubmodule=False, verbose=False):
    """Build scikit-tree using submodules.

    git submodule update --recursive --remote

    To update submodule wrt latest commits:

        git submodule update --init --recursive --remote
        git add -A
        git commit

    This will update the submodule, which then must be commited so that
    git knows the submodule needs to be at a certain commit hash.
    """
    ctx.invoke(setup_submodule, forcesubmodule=forcesubmodule)

    # run build as normal
    ctx.invoke(meson.build, meson_args=meson_args, jobs=jobs, clean=clean, verbose=verbose)


@cli.cls_cmd("bench")
class Bench(Task):
    """:wrench: Run benchmarks.

    \b
    ```python
     Examples:

    $ python dev.py bench -t integrate.SolveBVP
    $ python dev.py bench -t linalg.Norm
    $ python dev.py bench --compare main
    ```
    """

    ctx = CONTEXT
    TASK_META = {
        "task_dep": ["build"],
    }
    submodule = Option(
        ["--submodule", "-s"],
        default=None,
        metavar="SUBMODULE",
        help="Submodule whose tests to run (cluster, constants, ...)",
    )
    tests = Option(
        ["--tests", "-t"], default=None, multiple=True, metavar="TESTS", help="Specify tests to run"
    )
    compare = Option(
        ["--compare", "-c"],
        default=None,
        metavar="COMPARE",
        multiple=True,
        help=(
            "Compare benchmark results of current HEAD to BEFORE. "
            "Use an additional --bench COMMIT to override HEAD with COMMIT. "
            "Note that you need to commit your changes first!"
        ),
    )

    @staticmethod
    def run_asv(dirs, cmd):
        EXTRA_PATH = [
            "/usr/lib/ccache",
            "/usr/lib/f90cache",
            "/usr/local/lib/ccache",
            "/usr/local/lib/f90cache",
        ]
        bench_dir = dirs.root / "benchmarks"
        sys.path.insert(0, str(bench_dir))
        # Always use ccache, if installed
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(EXTRA_PATH + env.get("PATH", "").split(os.pathsep))
        # Control BLAS/LAPACK threads
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        # Limit memory usage
        from asv_benchmarks.benchmarks.common import set_mem_rlimit

        try:
            set_mem_rlimit()
        except (ImportError, RuntimeError):
            pass
        try:
            return subprocess.call(cmd, env=env, cwd=bench_dir)
        except OSError as err:
            if err.errno == errno.ENOENT:
                cmd_str = " ".join(cmd)
                print(f"Error when running '{cmd_str}': {err}\n")
                print(
                    "You need to install Airspeed Velocity "
                    "(https://airspeed-velocity.github.io/asv/)"
                )
                print("to run Scipy benchmarks")
                return 1
            raise

    @classmethod
    def scipy_bench(cls, args):
        dirs = Dirs(args)
        dirs.add_sys_path()
        print(f"SciPy from development installed path at: {dirs.site}")
        with working_dir(dirs.site):
            runner, version, mod_path = get_test_runner(PROJECT_MODULE)
            extra_argv = []
            if args.tests:
                extra_argv.append(args.tests)
            if args.submodule:
                extra_argv.append([args.submodule])

            bench_args = []
            for a in extra_argv:
                bench_args.extend(["--bench", " ".join(str(x) for x in a)])
            if not args.compare:
                print("Running benchmarks for Scipy version %s at %s" % (version, mod_path))
                cmd = [
                    "asv",
                    "run",
                    "--dry-run",
                    "--show-stderr",
                    "--python=same",
                    "--quick",
                ] + bench_args
                retval = cls.run_asv(dirs, cmd)
                sys.exit(retval)
            else:
                if len(args.compare) == 1:
                    commit_a = args.compare[0]
                    commit_b = "HEAD"
                elif len(args.compare) == 2:
                    commit_a, commit_b = args.compare
                else:
                    print("Too many commits to compare benchmarks for")
                # Check for uncommitted files
                if commit_b == "HEAD":
                    r1 = subprocess.call(["git", "diff-index", "--quiet", "--cached", "HEAD"])
                    r2 = subprocess.call(["git", "diff-files", "--quiet"])
                    if r1 != 0 or r2 != 0:
                        print("*" * 80)
                        print(
                            "WARNING: you have uncommitted changes --- "
                            "these will NOT be benchmarked!"
                        )
                        print("*" * 80)

                # Fix commit ids (HEAD is local to current repo)
                p = subprocess.Popen(["git", "rev-parse", commit_b], stdout=subprocess.PIPE)
                out, err = p.communicate()
                commit_b = out.strip()

                p = subprocess.Popen(["git", "rev-parse", commit_a], stdout=subprocess.PIPE)
                out, err = p.communicate()
                commit_a = out.strip()
                cmd_compare = [
                    "asv",
                    "continuous",
                    "--show-stderr",
                    "--factor",
                    "1.05",
                    "--quick",
                    commit_a,
                    commit_b,
                ] + bench_args
                cls.run_asv(dirs, cmd_compare)
                sys.exit(1)

    @classmethod
    def run(cls, **kwargs):
        """run benchmark"""
        kwargs.update(cls.ctx.get())
        Args = namedtuple("Args", [k for k in kwargs.keys()])
        args = Args(**kwargs)
        cls.scipy_bench(args)
