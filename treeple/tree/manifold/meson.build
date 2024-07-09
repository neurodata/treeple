tree_extension_metadata = {
  '_morf_splitter':
    {'sources': ['_morf_splitter.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
}

foreach ext_name, ext_dict : tree_extension_metadata
  py.extension_module(
    ext_name,
    ext_dict.get('sources'),
    dependencies: [np_dep],
    override_options : ext_dict.get('override_options', []),
    c_args: c_args,
    cython_args: cython_c_args,
    subdir: 'treeple/tree/manifold',
    install: true,
  )
endforeach


py.install_sources(
  subdir: 'treeple/tree/manifold'   # Folder relative to site-packages to install to
)
