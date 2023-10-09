project(
  'scikit-tree',
  'c', 'cpp',
  # Note that the git commit hash cannot be added dynamically here
  # That only happens when importing from a git repository.
  # See `sktree/__init__.py`
  version: '0.2.1',
  license: 'BSD-3',
  meson_version: '>= 0.64.0',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Check compiler is recent enough (see "Toolchain Roadmap" for details)
if cc.get_id() == 'gcc'
  if not cc.version().version_compare('>=8.0')
    error('scikit-tree requires GCC >= 8.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('scikit-tree requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for scikit-tree). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it. For Fortran code, Meson already adds `-lm`.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

cython = find_program(
  'cython',
  required: true
)
if not cython.found()
  error('MESON_BUILD_FAILED: Cython3 not found. Please install it.')
endif

# r = run_command('git', 'submodule', 'update', '--init', check: false)
r = run_command('mv', 'sktree/_lib/sklearn_fork/sklearn', 'sktree/_lib/sklearn', check: false)

# Setup Python:
# https://mesonbuild.com/Python-module.html
py3_mod = import('python')

# NOTE: with Meson >=0.64.0 we can add `pure: false` here and remove that line
# everywhere else, see https://github.com/mesonbuild/meson/pull/10783.
py3 = py3_mod.find_installation(
  'python3',
  pure: false  # Will be installed next to binaries
)
# py3.install_env('venv')

# print some debugging output
message(py3.full_path())
message(py3.get_install_dir())
if py3.language_version().version_compare('<3.9')
    error('At least Python 3.9 is required.')
endif
py3_dep = py3.dependency()

subdir('sktree')
