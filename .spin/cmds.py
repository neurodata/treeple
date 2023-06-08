import os
import shutil
import sys

import click
from spin import util
from spin.cmds import meson


import subprocess

def get_git_revision_hash(submodule) -> str:
    return subprocess.check_output(['git', 'rev-parse', submodule]).decode('ascii').strip()

def get_git_revision_short_hash(submodule) -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', submodule]).decode('ascii').strip()


@click.command()
@click.option("--build-dir", default="build", help="Build directory; default is `$PWD/build`")
@click.option("--clean", is_flag=True, help="Clean previously built docs before building")
def docs(build_dir, clean=False):
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

    os.environ["SPHINXOPTS"] = "-W"
    os.environ["PYTHONPATH"] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(["make", "-C", "docs", "clean", "html-noplot"], replace=True)


@click.command()
def coverage():
    """📊 Generate coverage report"""
    util.run(
        [
            "python",
            "-m",
            "spin",
            "test",
            "--",
            "-o",
            "python_functions=test_*",
            "sktree",
            "--cov=sktree",
            "--cov-report=xml",
        ],
        replace=True,
    )



@click.command()
@click.option("-j", "--jobs", help="Number of parallel tasks to launch", type=int)
@click.option("--clean", is_flag=True, help="Clean build directory before build")
@click.option(
    "-v", "--verbose", is_flag=True, help="Print all build output, even installation"
)
@click.argument("meson_args", nargs=-1)
def build(meson_args, jobs=None, clean=False, verbose=False):
    """📊 Generate coverage report"""
    import os
    
    commit_fpath = './sktree/_lib/commit.txt'
    submodule = './sktree/_lib/sklearn_fork'
    commit = ''
    current_hash = ''
    if os.path.exists(commit_fpath):
        with open(commit_fpath, 'r') as f:
            commit = f.read().strip()

        # get revision hash
        current_hash = get_git_revision_hash(submodule)
    else:
        util.run(
        [
            'git',
            'submodule',
            'update',
            '--init',
        ]
        )
    print(current_hash, commit)
    if current_hash == '' or current_hash != commit:
        util.run(
            [   
                'touch', './sktree/_lib/commit.txt',
            ],
        )
        util.run(['git', 'rev-parse', '@:./sktree/_lib/sklearn', '>', './sktree/_lib/commit.txt'])

        util.run(
            [
                'mv', 'sktree/_lib/sklearn/sklearn', 'sktree/_lib/sklearn_fork',
            ]
        )
        util.run(['rm', '-rf', 'sktree/_lib/sklearn/'])
        util.run(['mv', 'sktree/_lib/sklearn_fork', 'sktree/_lib/sklearn'])

    print('here...')
    meson.build(meson_args, jobs=jobs, clean=clean, verbose=verbose)
