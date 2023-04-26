extensions = [
    '_morf_splitter',
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen_cpp.process(ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy],
    install: true,
    subdir: 'sktree/tree/manifold',
  )
endforeach


py3.install_sources(
  subdir: 'sktree/tree/manifold'   # Folder relative to site-packages to install to
)
