/* NIXL plugin entry points. Loaded as libplugin_HETERO_NPU.so from NIXL_PLUGIN_DIR. */
#include "backend/backend_plugin.h"
#include "hetero_npu_backend.h"

namespace {
using plugin_t = nixlBackendPluginCreator<nixlHeteroNpuEngine>;
const char kName[] = "HETERO_NPU";
const char kVersion[] = "0.1.0";
}  // namespace

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *nixl_plugin_init() {
    return plugin_t::create(NIXL_PLUGIN_API_VERSION, kName, kVersion,
                            nixlHeteroNpuEngine::defaultParams(), {DRAM_SEG, VRAM_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {}
