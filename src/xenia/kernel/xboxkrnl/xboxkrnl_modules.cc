/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2019 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/logging.h"
#include "xenia/cpu/processor.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/xbox.h"

DECLARE_bool(mount_cache);

namespace xe {
namespace kernel {
namespace xboxkrnl {

dword_result_t XexCheckExecutablePrivilege(dword_t privilege) {
  // BOOL
  // DWORD Privilege

  // Privilege is bit position in xe_xex2_system_flags enum - so:
  // Privilege=6 -> 0x00000040 -> XEX_SYSTEM_INSECURE_SOCKETS
  uint32_t mask = 1 << (privilege % 32);

  auto module = kernel_state()->GetExecutableModule();
  if (!module) {
    return 0;
  }

  // Privilege 0xB (TitleInsecureUtilityDrive) hack:
  // If the 0xB privilege is set, the cache-partition code baked into most
  // games skips a huge chunk of device-init code (registering a custom
  // STFC filesystem handler with the kernel, etc), and just symlinks the
  // cache partition to the existing device directly (I guess on 360 this
  // would probably make it FATX, explaining the 'Insecure' part of it)

  // Thanks to this skip we can easily take control of the cache partition
  // ourselves, just by symlinking it before the game does!

  // TODO: check if this skip-code is actually available on every game that
  // uses cache - it's possible that early/later SDKs might not have it, and
  // we won't be able to rely on using this cheat for everything...
  if (cvars::mount_cache) {
    // Some games (eg. Sega Rally Revo) seem to have some code that checks for
    // privilege 0xB, and bails out if it's set - no idea why it does this,
    // maybe it's some kind of earlier STFC code, it seems to be ran just
    // before the real STFC code that this trick was made for is called.
    // We can use a cheeky heuristic to get past it tho: the bailout function
    // checks for 0xB and then checks 0x17, but the cache function that we're
    // trying to fool by setting 0xB always seems to check 0x17 before 0xB!

    // So if we check that we've seen 0x17 first before replying true to 0xB,
    // hopefully we can get passed the bailout function unscathed...

    // Privilege 0x17 (TitleBothUtilityPartitions)
    // TODO: probably should store this some other way, statics won't play well
    // when we have XEX re-loading etc...
    static bool seen_0x17 = false;
    if (privilege == 0x17) {
      seen_0x17 = true;
    } else if (privilege == 0xB && seen_0x17) {
      return 1;
    }
  }

  uint32_t header_id = XEX_HEADER_SYSTEM_FLAGS;  // header ID 0x30000
  // Privileges 32+ are stored in 0x30100, 64+ probably in 0x30200...
  header_id += (privilege / 32) << 8;

  uint32_t flags = 0;
  module->GetOptHeader<uint32_t>((xex2_header_keys)header_id, &flags);

  return (flags & mask) > 0;
}
DECLARE_XBOXKRNL_EXPORT1(XexCheckExecutablePrivilege, kModules, kImplemented);

dword_result_t XexGetModuleHandle(lpstring_t module_name,
                                  lpdword_t hmodule_ptr) {
  object_ref<XModule> module;

  if (!module_name) {
    module = kernel_state()->GetExecutableModule();
  } else {
    module = kernel_state()->GetModule(module_name);
  }

  if (!module) {
    *hmodule_ptr = 0;
    return X_ERROR_NOT_FOUND;
  }

  // NOTE: we don't retain the handle for return.
  *hmodule_ptr = module->hmodule_ptr();

  return X_ERROR_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XexGetModuleHandle, kModules, kImplemented);

dword_result_t XexGetModuleSection(lpvoid_t hmodule, lpstring_t name,
                                   lpdword_t data_ptr, lpdword_t size_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  auto module = XModule::GetFromHModule(kernel_state(), hmodule);
  if (module) {
    uint32_t section_data = 0;
    uint32_t section_size = 0;
    result = module->GetSection(name, &section_data, &section_size);
    if (XSUCCEEDED(result)) {
      *data_ptr = section_data;
      *size_ptr = section_size;
    }
  } else {
    result = X_STATUS_INVALID_HANDLE;
  }

  return result;
}
DECLARE_XBOXKRNL_EXPORT1(XexGetModuleSection, kModules, kImplemented);

dword_result_t XexLoadImage(lpstring_t module_name, dword_t module_flags,
                            dword_t min_version, lpdword_t hmodule_ptr) {
  X_STATUS result = X_STATUS_NO_SUCH_FILE;

  uint32_t hmodule = 0;
  auto module = kernel_state()->GetModule(module_name);
  if (module) {
    // Existing module found.
    hmodule = module->hmodule_ptr();
    result = X_STATUS_SUCCESS;
  } else {
    // Not found; attempt to load as a user module.
    auto user_module = kernel_state()->LoadUserModule(module_name);
    if (user_module) {
      user_module->Retain();
      hmodule = user_module->hmodule_ptr();
      result = X_STATUS_SUCCESS;
    }
  }

  // Increment the module's load count.
  if (hmodule) {
    auto ldr_data =
        kernel_memory()->TranslateVirtual<X_LDR_DATA_TABLE_ENTRY*>(hmodule);
    ldr_data->load_count++;
  }

  *hmodule_ptr = hmodule;

  return result;
}
DECLARE_XBOXKRNL_EXPORT1(XexLoadImage, kModules, kImplemented);

dword_result_t XexUnloadImage(lpvoid_t hmodule) {
  auto module = XModule::GetFromHModule(kernel_state(), hmodule);
  if (!module) {
    return X_STATUS_INVALID_HANDLE;
  }

  // Can't unload kernel modules from user code.
  if (module->module_type() != XModule::ModuleType::kKernelModule) {
    auto ldr_data = hmodule.as<X_LDR_DATA_TABLE_ENTRY*>();
    if (--ldr_data->load_count == 0) {
      // No more references, free it.
      module->Release();
      kernel_state()->UnloadUserModule(object_ref<UserModule>(
          reinterpret_cast<UserModule*>(module.release())));
    }
  }

  return X_STATUS_SUCCESS;
}
DECLARE_XBOXKRNL_EXPORT1(XexUnloadImage, kModules, kImplemented);

dword_result_t XexGetProcedureAddress(lpvoid_t hmodule, dword_t ordinal,
                                      lpdword_t out_function_ptr) {
  // May be entry point?
  assert_not_zero(ordinal);

  bool is_string_name = (ordinal & 0xFFFF0000) != 0;
  auto string_name =
      reinterpret_cast<const char*>(kernel_memory()->TranslateVirtual(ordinal));

  X_STATUS result = X_STATUS_INVALID_HANDLE;

  object_ref<XModule> module;
  if (!hmodule) {
    module = kernel_state()->GetExecutableModule();
  } else {
    module = XModule::GetFromHModule(kernel_state(), hmodule);
  }
  if (module) {
    uint32_t ptr;
    if (is_string_name) {
      ptr = module->GetProcAddressByName(string_name);
    } else {
      ptr = module->GetProcAddressByOrdinal(ordinal);
    }
    if (ptr) {
      *out_function_ptr = ptr;
      result = X_STATUS_SUCCESS;
    } else {
      XELOGW("ERROR: XexGetProcedureAddress ordinal not found!");
      *out_function_ptr = 0;
      result = X_STATUS_DRIVER_ENTRYPOINT_NOT_FOUND;
    }
  }

  return result;
}
DECLARE_XBOXKRNL_EXPORT1(XexGetProcedureAddress, kModules, kImplemented);

void ExRegisterTitleTerminateNotification(
    pointer_t<X_EX_TITLE_TERMINATE_REGISTRATION> reg, dword_t create) {
  if (create) {
    // Adding.
    kernel_state()->RegisterTitleTerminateNotification(
        reg->notification_routine, reg->priority);
  } else {
    // Removing.
    kernel_state()->RemoveTitleTerminateNotification(reg->notification_routine);
  }
}
DECLARE_XBOXKRNL_EXPORT1(ExRegisterTitleTerminateNotification, kModules,
                         kImplemented);

void RegisterModuleExports(xe::cpu::ExportResolver* export_resolver,
                           KernelState* kernel_state) {}

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe
