extensions = [
  '_unsup_criterion'
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy],
    install: true,
    subdir: 'sktree/tree'
  )
endforeach

# TODO: comment in _classes.py when we have a working Cython unsupervised tree with a Python API
python_sources = [
  '__init__.py',
#   '_classes.py',
]

py3.install_sources(
  python_sources,
  pure: false,            # Will be installed next to binaries
  subdir: 'sktree/tree'   # Folder relative to site-packages to install to
)

# TODO: comment in if we include tests
# subdir('tests')