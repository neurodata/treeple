project(
  'treeple',
  'c', 'cpp', 'cython',
  # Note that the git commit hash cannot be added dynamically here
  # That only happens when importing from a git repository.
  # See `treeple/__init__.py`
  version: '0.9.1',
  license: 'PolyForm Noncommercial 1.0.0',
  meson_version: '>= 1.1.0',
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
    error('treeple requires GCC >= 8.0')
  endif
elif cc.get_id() == 'msvc'
  if not cc.version().version_compare('>=19.20')
    error('treeple requires at least vc142 (default with Visual Studio 2019) ' + \
          'when building with MSVC')
  endif
endif

# Suppress warning for deprecated Numpy API.
# Replace with numpy_nodepr_api after Cython 3.0 is out
# '-Wno-maybe-uninitialized'
# numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'

# (Suppress warning messages emitted by #warning directives).
_global_c_args = cc.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
)
add_project_arguments(_global_c_args, language : 'c')

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for treeple). For C++ it isn't needed, because libstdc++/libc++ is
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
r = run_command('mv', 'treeple/_lib/sklearn_fork/sklearn', 'treeple/_lib/sklearn', check: false)

# Setup Python:
# https://mesonbuild.com/Python-module.html
py = import('python').find_installation(pure: false)

# print some debugging output
message(py.full_path())
message(py.get_install_dir())
if py.language_version().version_compare('<3.9')
    error('At least Python 3.9 is required.')
endif

subdir('treeple')
