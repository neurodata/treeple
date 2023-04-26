extensions = [
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
    install: true,
    subdir: 'sktree/tree/unsupervised',
  )
endforeach


py3.install_sources(
  subdir: 'sktree/tree/unsupervised'   # Folder relative to site-packages to install to
)
