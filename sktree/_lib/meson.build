fs = import('fs')
if not fs.exists('sklearn/README.rst')
  error('Missing the `sklearn` fork submodule! Run `git submodule update --init` to fix this.')
endif

# install _lib/
python_sources = [
  '__init__.py',
  'test.py',
]
py3.install_sources(
  python_sources,
  subdir: 'sktree/_lib'
)

# install tree/ submodule
extensions = [
  '_utils',
  '_splitter',
  '_tree',
  '_criterion',
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen_cpp.process('./sklearn/sklearn/tree/' + ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy,],
    install: true,
    subdir: 'sktree/_lib/sklearn/tree/',
  )
endforeach

python_sources = [
  '_classes.py',
]
foreach py_source: python_sources
  py3.install_sources(
    './sklearn/sklearn/tree/' + py_source,
    subdir: 'sktree/_lib/sklearn/tree/'
  )
endforeach

# install ensemble/ submodule
python_sources = [
  '_forest.py',
]
foreach py_source: python_sources
  py3.install_sources(
    './sklearn/sklearn/ensemble/' + py_source,
    subdir: 'sktree/_lib/sklearn/ensemble'
  )
endforeach
