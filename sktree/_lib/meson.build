fs = import('fs')
if not fs.exists('sklearn/README.rst')
  error('Missing the `sklearn` fork submodule! Run `git submodule update --init` to fix this.')
endif

python_sources = [
  '__init__.py',
]

py3.install_sources(
  python_sources,
  subdir: 'sktree/_lib'
)

subdir('sklearn')
