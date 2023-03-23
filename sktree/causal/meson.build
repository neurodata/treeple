extensions = [
  '_grf_criterion',
  '_splitter',
  '_utils',
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen_cpp.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy],
    install: true,
    subdir: 'sktree/causal',
  )
endforeach

python_sources = [
  '__init__.py',
  'tree.py',
  'meta.py',
]

py3.install_sources(
  python_sources,
  subdir: 'sktree/causal'   # Folder relative to site-packages to install to
)

# TODO: comment in if we include tests
subdir('tests')