tree_extension_metadata = {
  '_unsup_criterion':
    {'sources': ['_unsup_criterion.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_unsup_splitter':
    {'sources': ['_unsup_splitter.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_unsup_tree':
    {'sources': ['_unsup_tree.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_unsup_oblique_splitter':
    {'sources': ['_unsup_oblique_splitter.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_unsup_oblique_tree':
    {'sources': ['_unsup_oblique_tree.pyx'],
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
    subdir: 'treeple/tree/unsupervised',
    install: true,
  )
endforeach


py.install_sources(
  subdir: 'treeple/tree/unsupervised'   # Folder relative to site-packages to install to
)
