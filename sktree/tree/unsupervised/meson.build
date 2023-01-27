extensions = [
  '_unsup_criterion',
  '_unsup_splitter',
  '_unsup_tree',
  '_unsup_oblique_splitter',
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

