fs = import('fs')
if not fs.exists('sklearn')
  error('Missing the `sklearn` fork submodule! Run `git submodule update --init` to fix this.')
endif

# install tree/ submodule
extensions = [
  '_tree',
  '_utils',
  '_criterion',
  '_splitter',
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen_cpp.process('./sklearn/tree/' + ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy,],
    install: true,
    subdir: 'sktree/_lib/sklearn/tree/',
  )
endforeach

python_sources = [
  './sklearn/tree/__init__.py',
  './sklearn/tree/_classes.py',
  './sklearn/tree/_export.py',
  './sklearn/tree/_reingold_tilford.py',
]

py3.install_sources(
  python_sources,
  subdir: 'sktree/_lib/sklearn/tree'   # Folder relative to site-packages to install to
)

# install ensemble/ submodule
python_sources = [
  '_forest.py',
]
foreach py_source: python_sources
  py3.install_sources(
    './sklearn/ensemble/' + py_source,
    subdir: 'sktree/_lib/sklearn/ensemble'
  )
endforeach