extensions = [
  '_sklearn_splitter',
  '_unsup_criterion',
  '_unsup_splitter',
  '_unsup_tree',
  '_unsup_oblique_splitter',
  '_unsup_oblique_tree',
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen_cpp.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy],
    # override_options : ['cython_language=cpp'],
    install: true,
    subdir: 'sktree/tree'
  )
endforeach

# TODO: comment in _classes.py when we have a working Cython unsupervised tree with a Python API
python_sources = [
  '__init__.py',
  '_classes.py',
]

py3.install_sources(
  python_sources,
  subdir: 'sktree/tree'   # Folder relative to site-packages to install to
)

# TODO: comment in if we include tests
subdir('tests')