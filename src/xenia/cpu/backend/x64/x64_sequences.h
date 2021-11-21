/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2014 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_CPU_BACKEND_X64_X64_SEQUENCES_H_
#define XENIA_CPU_BACKEND_X64_X64_SEQUENCES_H_

#include "xenia/cpu/hir/instr.h"

#include <unordered_map>

#include <algorithm>
#include <cstring>

#include "xenia/cpu/backend/x64/x64_op.h"
#include "xenia/cpu/backend/x64/x64_tracers.h"

namespace xe {
namespace cpu {
namespace backend {
namespace x64 {

class X64Emitter;

typedef bool (*SequenceSelectFn)(X64Emitter&, const hir::Instr*);
extern std::unordered_map<uint32_t, SequenceSelectFn> sequence_table;

template <typename T>
bool Register() {
  sequence_table.insert({T::head_key(), T::Select});
  return true;
}

template <typename T, typename Tn, typename... Ts>
static bool Register() {
  bool b = true;
  b = b && Register<T>();          // Call the above function
  b = b && Register<Tn, Ts...>();  // Call ourself again (recursively)
  return b;
}
#define EMITTER_OPCODE_TABLE(name, ...) \
  const auto X64_INSTR_##name = Register<__VA_ARGS__>();

bool SelectSequence(X64Emitter* e, const hir::Instr* i,
                    const hir::Instr** new_tail);

bool SelectMultiSequence(X64Emitter* e, const hir::Instr* i, const hir::Instr** new_tail);


template <typename T>
RegExp ComputeMemoryAddressOffset(X64Emitter& e, const T& guest,
                                  const T& offset) {
  assert_true(offset.is_constant);
  int32_t offset_const = static_cast<int32_t>(offset.constant());

  if (guest.is_constant) {
    uint32_t address = static_cast<uint32_t>(guest.constant());
    address += offset_const;
    if (address < 0x80000000) {
      return e.GetMembaseReg() + address;
    } else {
      e.mov(e.eax, address);
      return e.GetMembaseReg() + e.rax;
    }
  } else {
    // Clear the top 32 bits, as they are likely garbage.
    // TODO(benvanik): find a way to avoid doing this.
    e.mov(e.eax, guest.reg().cvt32());
    return e.GetMembaseReg() + e.rax + offset_const;
  }
}

template<typename T>
void translate_address_in_register(X64Emitter& e, T& reg)  {
    e.add(reg, e.GetMembaseReg());
}

template<typename T>
void translate_address_in_register(Xbyak::CodeGenerator& e, T& reg)  {
    e.add(reg, e.rdi);
}
// Note: most *should* be aligned, but needs to be checked!
template <typename T>
RegExp ComputeMemoryAddress(X64Emitter& e, const T& guest) {
  if (guest.is_constant) {
    // TODO(benvanik): figure out how to do this without a temp.
    // Since the constant is often 0x8... if we tried to use that as a
    // displacement it would be sign extended and mess things up.
    uint32_t address = static_cast<uint32_t>(guest.constant());
    if (address < 0x80000000) {
      return e.GetMembaseReg() + address;
    } else {
      e.mov(e.eax, address);
      return e.GetMembaseReg() + e.rax;
    }
  } else {
    // Clear the top 32 bits, as they are likely garbage.
    // TODO(benvanik): find a way to avoid doing this.
    e.mov(e.eax, guest.reg().cvt32());
    return e.GetMembaseReg() + e.rax;
  }
}

}  // namespace x64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_X64_X64_SEQUENCES_H_
