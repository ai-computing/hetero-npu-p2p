# HETERO_NPU NIXL backend plugin (docs/nixl-backend.md).
#
# Out-of-tree build against an installed NIXL (pip wheel or source install). Plugins must use the
# same C++ standard and libstdc++ ABI as NIXL core, and the headers of the same NIXL tag.
#   meson setup build -Dnixl_src=... -Dnixl_libdir=... -Dabsl_include=... -Dliburing_root=...
#   ninja -C build            # -> build/libplugin_HETERO_NPU.so
# In-tree alternative: copy this directory to <nixl>/src/plugins/hetero_npu and add a subdir().
project('hetero-npu-nixl-plugin', 'cpp', version: '0.1.0',
        default_options: ['cpp_std=c++20', 'buildtype=release', 'warning_level=2'])

nixl_src = get_option('nixl_src')
if nixl_src == ''
  error('-Dnixl_src=<NIXL source tree> is required (headers of the runtime NIXL version)')
endif
nixl_inc = include_directories(
  nixl_src / 'src/api/cpp',
  nixl_src / 'src/api/cpp/backend',
  nixl_src / 'src/api/cpp/telemetry',
  nixl_src / 'src/core/telemetry',
  nixl_src / 'src/infra',
  nixl_src / 'src/utils',
  is_system: true)

cxx = meson.get_compiler('cpp')
deps = []
if get_option('absl_include') != ''
  absl_inc = include_directories(get_option('absl_include'), is_system: true)
else
  absl_inc = []
  deps += dependency('absl_log')
endif

uring_root = get_option('liburing_root')
if uring_root != ''
  # Static, as native/build.py links the helper: no liburing.so.2 needed at runtime.
  uring_inc = include_directories(uring_root / 'src/include', is_system: true)
  deps += declare_dependency(link_args: [uring_root / 'src/liburing.a'])
else
  uring_inc = []
  deps += dependency('liburing')
endif
rpaths = []

nixl_libdir = get_option('nixl_libdir')
if nixl_libdir != ''
  deps += [cxx.find_library('nixl_build', dirs: [nixl_libdir]),
           cxx.find_library('nixl_common', dirs: [nixl_libdir])]
  rpaths += [nixl_libdir]
else
  deps += dependency('nixl')
endif

driver_inc = include_directories(get_option('driver_root') / 'include',
                                 get_option('driver_root') / 'src', is_system: true)

plugin = shared_library('HETERO_NPU',
  ['hetero_npu_plugin.cpp', 'hetero_npu_backend.cpp', 'pdma_queue.cpp'],
  name_prefix: 'libplugin_',
  include_directories: [nixl_inc, absl_inc, uring_inc, driver_inc],
  dependencies: deps,
  build_rpath: ':'.join(rpaths),
  install: true,
  install_rpath: ':'.join(rpaths))
