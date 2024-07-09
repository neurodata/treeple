fs = import('fs')
if not fs.exists('sklearn')
  error('Missing the `sklearn` fork submodule! Run `git submodule update --init` to fix this.')
endif

# install tree/ submodule
tree_extension_metadata = {
  '_tree':
    {'sources': ['./sklearn/tree/' + '_tree.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_splitter':
    {'sources': ['./sklearn/tree/' + '_splitter.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_criterion':
    {'sources': ['./sklearn/tree/' + '_criterion.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
  '_utils':
    {'sources': ['./sklearn/tree/' + '_utils.pyx'],
     'override_options': ['cython_language=cpp', 'optimization=3']},
}


foreach ext_name, ext_dict : tree_extension_metadata
  py.extension_module(
    ext_name,
    ext_dict.get('sources'),
    dependencies: [np_dep],
    override_options : ext_dict.get('override_options', []),
    cython_args: cython_c_args,
    subdir: 'treeple/_lib/sklearn/tree/',
    install: true
  )
endforeach

python_sources = [
  './sklearn/tree/__init__.py',
  './sklearn/tree/_classes.py',
  './sklearn/tree/_export.py',
  './sklearn/tree/_reingold_tilford.py',
]

py.install_sources(
  python_sources,
  subdir: 'treeple/_lib/sklearn/tree'   # Folder relative to site-packages to install to
)

# install ensemble/ submodule
python_sources = [
  '_forest.py',
]
foreach py_source: python_sources
  py.install_sources(
    './sklearn/ensemble/' + py_source,
    subdir: 'treeple/_lib/sklearn/ensemble'
  )
endforeach

# TODO: Can remove if included in scikit-learn eventually
# install tree/ submodule
extensions = [
  '_quad_tree',
]

foreach ext: extensions
  py.extension_module(
    ext,
    ['./sklearn/neighbors/' + ext + '.pyx'],
    c_args: c_args,
    dependencies: [np_dep],
    cython_args: cython_c_args,
    override_options : ['optimization=3', 'cython_language=cpp'],
    install: true,
    subdir: 'treeple/_lib/sklearn/neighbors/',
  )
endforeach

# install tree/ submodule
extensions = [
  '_typedefs',
  '_random',
]

foreach ext: extensions
  py.extension_module(ext,
    ['./sklearn/utils/' + ext + '.pyx'],
    c_args: c_args,
    dependencies: [np_dep],
    cython_args: cython_c_args,
    override_options : ['optimization=3', 'cython_language=cpp'],
    install: true,
    subdir: 'treeple/_lib/sklearn/utils/',
  )
endforeach
