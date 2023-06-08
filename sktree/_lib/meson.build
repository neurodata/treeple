# fs = import('fs')
# if not fs.exists('sklearn/README.rst')
#   error('Missing the `sklearn` fork submodule! Run `git submodule update --init` to fix this.')
# endif

# install tree/ submodule
extensions = [
  '_tree',
  '_utils',
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

# install test file
extensions = [
  'stuff',
]

foreach ext: extensions
  py3.extension_module(ext,
    cython_gen_cpp.process('./' + ext + '.pyx'),
    c_args: cython_c_args,
    include_directories: [incdir_numpy,],
    install: true,
    subdir: 'sktree/_lib/',
  )
endforeach
