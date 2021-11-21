/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2018 Xenia Developers. All rights reserved.                      *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <array>
#include <bitset>
#include <cstring>

#include "xenia/cpu/backend/x64/x64_code_cache.h"
#include "xenia/cpu/backend/x64/x64_emitter.h"
#include "xenia/cpu/backend/x64/x64_op.h"
#include "xenia/cpu/backend/x64/x64_sequences.h"
#include "xenia/cpu/backend/x64/x64_stack_layout.h"
#include "xenia/cpu/processor.h"
#include "xenia/cpu/xex_module.h"
namespace xe {
namespace cpu {
namespace backend {
namespace x64 {

const Instr* hunt_context_store_instr_backwards(const Instr* from,
                                                unsigned offset,
                                                bool* notinblock) {
  const Instr* pos = from->prev;

  while (pos) {
    if (pos->opcode == &OPCODE_STORE_CONTEXT_info) {
      /*
        i->src1.offset = offset;
        i->set_src2(value);
        i->src3.value = NULL;
      */

      if (pos->src1.offset == offset) {
        return pos;
      }
    } else if (pos->opcode->flags & OPCODE_FLAG_VOLATILE) {
      *notinblock = false;
      return nullptr;
    }
    pos = pos->prev;
  }
  *notinblock = true;
  return nullptr;
}
const Value* hunt_context_store_backwards(const Instr* from, unsigned offset) {
  const Instr* pos = from->prev;
  while (pos) {
    if (pos->opcode == &OPCODE_STORE_CONTEXT_info) {
      /*
        i->src1.offset = offset;
        i->set_src2(value);
        i->src3.value = NULL;
      */

      if (pos->src1.offset == offset) {
        return pos->src2.value;
      }
    } else if (pos->opcode->flags & OPCODE_FLAG_VOLATILE) {
      return nullptr;
    }
    pos = pos->prev;
  }
  return nullptr;
}

uintptr_t translate_address_compiletime(X64Emitter& e, uint32_t addr) {
  return (uintptr_t)(e.processor()->memory()->TranslateVirtual(addr));
}
class builtin_emitter_t;

/*
    helper class for builtin generation
    provides common methods and manages register allocation for simpler
    sequence design
*/
class builtin_emitter_t {
  X64Emitter* e;

  const Instr* m_current_instruction;

  std::vector<Xbyak::Reg64> m_free_gpregs;
  std::vector<Xbyak::Xmm> m_free_simdregs;

  // set of ppc gp registers that we can use the locations of
  // for storing temporaries
  // because of how often these are accessed, we can rely on them pretty much
  // always being in L1 cache so accesses are very fast
  std::bitset<32> m_available_ppc_gpreg_slots;
  std::bitset<32> m_available_fpureg_slots;
  std::bitset<128> m_available_vmxreg_slots;

 public:
  builtin_emitter_t(X64Emitter& _e, const Instr* current_instruction)
      : e(&_e),
        m_current_instruction(current_instruction),
        m_free_gpregs(),
        m_free_simdregs(),
        m_available_ppc_gpreg_slots() {
    default_state();
  }

  inline X64Emitter* operator->() { return e; }

  const Value* find_context_store_back(unsigned offs) {
    return hunt_context_store_backwards(m_current_instruction, offs);
  }
  /*
    if the register allocator allocated the value currently in context_offset
    to an x86 reg, find the reg
  */
  const Xbyak::Reg find_preloaded_gpreg(unsigned context_offset, TypeName typ,
                                        bool* success) {
    bool notinblock;
    auto idef = hunt_context_store_instr_backwards(m_current_instruction,
                                                   context_offset, &notinblock);

    if (!idef && notinblock) {
      for (auto pos = m_current_instruction->prev; pos; pos = pos->prev) {
        if (pos->opcode == &OPCODE_LOAD_info && pos->dest->type == typ &&
            pos->dest->reg.set &&
            (pos->dest->reg.set->types & MachineInfo::RegisterSet::INT_TYPES)) {
          for (auto pos2 = pos->next; pos2 != m_current_instruction;
               pos2 = pos2->next) {
            if (!pos2->dest) continue;
            if (pos2->dest->reg.index == pos->dest->reg.index &&
                (pos2->dest->reg.set->types &
                 MachineInfo::RegisterSet::INT_TYPES)) {
              *success = false;
              return e->eax;
            }
          }

          Xbyak::Reg64 result;
          e->SetupReg(pos->dest, result);
          *success = true;
          return result;
        }
      }
    } else if (idef) {
      for (auto pos = m_current_instruction->prev; pos != idef;
           pos = pos->prev) {
        if (pos->opcode == &OPCODE_LOAD_info && pos->dest->type == typ &&
            pos->dest->reg.set &&
            (pos->dest->reg.set->types & MachineInfo::RegisterSet::INT_TYPES)) {
          for (auto pos2 = pos->next; pos2 != m_current_instruction;
               pos2 = pos2->next) {
            if (!pos2->dest) continue;
            if (pos2->dest->reg.index == pos->dest->reg.index &&
                (pos2->dest->reg.set->types &
                 MachineInfo::RegisterSet::INT_TYPES)) {
              *success = false;
              return e->eax;
            }
          }
          Xbyak::Reg64 result;
          e->SetupReg(pos->dest, result);
          *success = true;
          return result;
        }
      }
    }
    *success = false;
    return e->rax;
  }

  void load_gpreg_arg(unsigned whichreg, TypeName typ, const Xbyak::Reg& into) {
    bool found_preld;
    auto preloaded = find_preloaded_gpreg(
        offsetof(ppc::PPCContext_s, r[whichreg]), typ, &found_preld);
    if (!found_preld) {
      const Value* ctxstore =
          find_context_store_back(offsetof(ppc::PPCContext_s, r[whichreg]));

      if (ctxstore && ctxstore->IsConstant()) {
        e->mov(into, ctxstore->constant.u64);
      } else {
        e->mov(into, e->ptr[e->GetContextReg() +
                            offsetof(ppc::PPCContext_s, r[whichreg])]);
      }
    } else {
      e->mov(into, preloaded);
    }
  }
  // if you used the membase reg in a scope that didnt need to address guest
  // memory/already translated ptrs call this to reload from context
  void reload_membase() {
    e->mov(e->GetMembaseReg(),
           e->ptr[e->GetContextReg() +
                  offsetof(ppc::PPCContext_s, virtual_membase)]);
  }

  template <unsigned... gps>
  void set_avail_gps() {
    constexpr unsigned avails[] = {gps...};
    for (auto&& avail : avails) {
      m_available_ppc_gpreg_slots.set(avail);
    }
  }

  unsigned alloc_ppc_gpreg() {
    uint32_t res = 0;
    bool r = xe::bit_scan_forward(
        (unsigned)m_available_ppc_gpreg_slots.to_ulong(), &res);

    assert_always(r);
    m_available_ppc_gpreg_slots.reset(res);
    return res;
  }

  void free_ppc_gpreg(unsigned reg) {
    assert_always(!m_available_ppc_gpreg_slots.test(reg));
    m_available_ppc_gpreg_slots.set(reg);
  }

  void default_state() {
    Xbyak::Reg64 rs[] = {e->rbp, e->rax, e->rcx, e->rdx, e->r8, e->r9};

    Xbyak::Xmm xs[] = {e->xmm0, e->xmm1, e->xmm2, e->xmm3};

    for (auto&& r : rs) {
      m_free_gpregs.push_back(r);
    }

    for (auto&& x : xs) m_free_simdregs.push_back(x);

    m_available_ppc_gpreg_slots.reset();
  }

  inline Xbyak::Reg64 allocate_gp() {
    if (m_free_gpregs.size() == 0) {
      assert_always(false && "out of gp registers!! ");
    }
    Xbyak::Reg64 r = m_free_gpregs.back();
    m_free_gpregs.pop_back();
    return r;
  }

  inline void release_gp(Xbyak::Reg64 r) {
    for (auto&& rr : m_free_gpregs) {
      if (rr == r) {
        assert_always(false && "register already available in gp set");
      }
    }
    m_free_gpregs.push_back(r);
  }

  inline void testjnz(Xbyak::Reg x, Xbyak::Label& to) {
    e->test(x, x);
    e->jnz(to);
  }

  inline void testjz(Xbyak::Reg x, Xbyak::Label& to) {
    e->test(x, x);
    e->jnz(to);
  }

  /*
     allocate a reg whose <64 parts can be accessed without a rex prefix
  */
  inline Xbyak::Reg64 allocate_nonrex_gp() {
    for (auto r = m_free_gpregs.begin(); r != m_free_gpregs.end(); ++r) {
      if (r->getIdx() < e->r8.getIdx()) {
        Xbyak::Reg64 result = *r;
        m_free_gpregs.erase(r);
        return result;
      }
    }
    assert_always(false && "out of gp registers!! no nonrex avail");
  }

  inline Xbyak::Xmm allocate_xmm() {
    assert_always(m_free_simdregs.size() != 0);
    Xbyak::Xmm result = m_free_simdregs.back();
    m_free_simdregs.pop_back();
    return result;
  }
  template <typename TCallableGen>
  size_t new_named_function(const char* name, TCallableGen&& implgenerator) {
    uint32_t cr = e->processor()->memory()->locate_named_memory_region(name);

    if (cr) {
      size_t result = static_cast<size_t>(cr) & 0xFFFFFFFFULL;
      return result;
    } else {
      Xbyak::CodeGenerator cgen{};

      implgenerator(cgen);
      const uint8_t* code = cgen.getCode();
      unsigned sz = cgen.getSize();

      uint32_t loc = e->processor()->memory()->find_or_init_named_memory_region(
          name, const_cast<uint8_t*>(code), sz,
          memory::PageAccess::kExecuteReadWrite);

      return static_cast<size_t>(loc) & 0xFFFFFFFFULL;
    }
  }
  template <typename TCallableGen>
  void call_named_function(const char* name, TCallableGen&& implgenerator) {
    size_t offs = new_named_function(name, implgenerator);

    e->mov(e->rax, offs);
    e->call(e->rax);
  }
};

#define PPCFIELD_PTR_BLTIN(eee) \
  e.ptr[ee->GetContextReg() + offsetof(ppc::PPCContext_s, eee)]

volatile int anchor_control = 0;
template <typename T>
static void EmitFusedBranch(X64Emitter& e, const T& i) {
  bool valid = i.instr->prev && i.instr->prev->dest == i.src1.value;
  auto opcode = valid ? i.instr->prev->opcode->num : -1;
  if (valid) {
    auto name = i.src2.value->name;
    switch (opcode) {
      case OPCODE_IS_TRUE:
        e.jnz(name, e.T_NEAR);
        break;
      case OPCODE_IS_FALSE:
        e.jz(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_EQ:
        e.je(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_NE:
        e.jne(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_SLT:
        e.jl(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_SLE:
        e.jle(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_SGT:
        e.jg(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_SGE:
        e.jge(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_ULT:
        e.jb(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_ULE:
        e.jbe(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_UGT:
        e.ja(name, e.T_NEAR);
        break;
      case OPCODE_COMPARE_UGE:
        e.jae(name, e.T_NEAR);
        break;
      default:
        e.test(i.src1, i.src1);
        e.jnz(name, e.T_NEAR);
        break;
    }
  } else {
    e.test(i.src1, i.src1);
    e.jnz(i.src2.value->name, e.T_NEAR);
  }
}
// ============================================================================
// OPCODE_DEBUG_BREAK
// ============================================================================
struct DEBUG_BREAK : Sequence<DEBUG_BREAK, I<OPCODE_DEBUG_BREAK, VoidOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) { e.DebugBreak(); }
};
EMITTER_OPCODE_TABLE(OPCODE_DEBUG_BREAK, DEBUG_BREAK);

// ============================================================================
// OPCODE_DEBUG_BREAK_TRUE
// ============================================================================
struct DEBUG_BREAK_TRUE_I8
    : Sequence<DEBUG_BREAK_TRUE_I8, I<OPCODE_DEBUG_BREAK_TRUE, VoidOp, I8Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.DebugBreak();
    e.L(skip);
  }
};
struct DEBUG_BREAK_TRUE_I16
    : Sequence<DEBUG_BREAK_TRUE_I16,
               I<OPCODE_DEBUG_BREAK_TRUE, VoidOp, I16Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.DebugBreak();
    e.L(skip);
  }
};
struct DEBUG_BREAK_TRUE_I32
    : Sequence<DEBUG_BREAK_TRUE_I32,
               I<OPCODE_DEBUG_BREAK_TRUE, VoidOp, I32Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.DebugBreak();
    e.L(skip);
  }
};
struct DEBUG_BREAK_TRUE_I64
    : Sequence<DEBUG_BREAK_TRUE_I64,
               I<OPCODE_DEBUG_BREAK_TRUE, VoidOp, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.DebugBreak();
    e.L(skip);
  }
};
struct DEBUG_BREAK_TRUE_F32
    : Sequence<DEBUG_BREAK_TRUE_F32,
               I<OPCODE_DEBUG_BREAK_TRUE, VoidOp, F32Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.DebugBreak();
    e.L(skip);
  }
};
struct DEBUG_BREAK_TRUE_F64
    : Sequence<DEBUG_BREAK_TRUE_F64,
               I<OPCODE_DEBUG_BREAK_TRUE, VoidOp, F64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.DebugBreak();
    e.L(skip);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_DEBUG_BREAK_TRUE, DEBUG_BREAK_TRUE_I8,
                     DEBUG_BREAK_TRUE_I16, DEBUG_BREAK_TRUE_I32,
                     DEBUG_BREAK_TRUE_I64, DEBUG_BREAK_TRUE_F32,
                     DEBUG_BREAK_TRUE_F64);

// ============================================================================
// OPCODE_TRAP
// ============================================================================
struct TRAP : Sequence<TRAP, I<OPCODE_TRAP, VoidOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.Trap(i.instr->flags);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_TRAP, TRAP);

// ============================================================================
// OPCODE_TRAP_TRUE
// ============================================================================
struct TRAP_TRUE_I8
    : Sequence<TRAP_TRUE_I8, I<OPCODE_TRAP_TRUE, VoidOp, I8Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.Trap(i.instr->flags);
    e.L(skip);
  }
};
struct TRAP_TRUE_I16
    : Sequence<TRAP_TRUE_I16, I<OPCODE_TRAP_TRUE, VoidOp, I16Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.Trap(i.instr->flags);
    e.L(skip);
  }
};
struct TRAP_TRUE_I32
    : Sequence<TRAP_TRUE_I32, I<OPCODE_TRAP_TRUE, VoidOp, I32Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.Trap(i.instr->flags);
    e.L(skip);
  }
};
struct TRAP_TRUE_I64
    : Sequence<TRAP_TRUE_I64, I<OPCODE_TRAP_TRUE, VoidOp, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.Trap(i.instr->flags);
    e.L(skip);
  }
};
struct TRAP_TRUE_F32
    : Sequence<TRAP_TRUE_F32, I<OPCODE_TRAP_TRUE, VoidOp, F32Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.Trap(i.instr->flags);
    e.L(skip);
  }
};
struct TRAP_TRUE_F64
    : Sequence<TRAP_TRUE_F64, I<OPCODE_TRAP_TRUE, VoidOp, F64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    e.Trap(i.instr->flags);
    e.L(skip);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_TRAP_TRUE, TRAP_TRUE_I8, TRAP_TRUE_I16,
                     TRAP_TRUE_I32, TRAP_TRUE_I64, TRAP_TRUE_F32,
                     TRAP_TRUE_F64);

static float load_be_f(unsigned* where) {
  unsigned heya = _load_be_u32((const void*)where);
  return *reinterpret_cast<float*>(&heya);
}

static void sto_be_f(unsigned* where, float f) {
  unsigned ayy = *reinterpret_cast<unsigned*>(&f);

  _store_be_u32(where, ayy);
}
/*
 sub         rsp,18h
    __debugbreak();
 int         3
  float x = load_be_f(vec);
 movbe       eax,dword ptr [r8]
 mov         dword ptr [rsp+30h],eax
  float y = load_be_f(&vec[1]);
 movbe       eax,dword ptr [r8+4]


  float sq = sqrtf((x * x) + (y * y) + (z * z));
 vmovss      xmm1,dword ptr [x]
  float y = load_be_f(&vec[1]);
 mov         dword ptr [rsp+38h],eax
  float z = load_be_f(&vec[2]);
 movbe       eax,dword ptr [r8+8]


  float sq = sqrtf((x * x) + (y * y) + (z * z));
 vmovss      xmm3,dword ptr [y]
 vmulss      xmm0,xmm1,xmm1
 vfmadd231ss xmm0,xmm3,xmm3
  float z = load_be_f(&vec[2]);
 mov         dword ptr [rsp],eax
 vmovss      xmm4,dword ptr [rsp]
 vfmadd231ss xmm0,xmm4,xmm4
 vsqrtss     xmm2,xmm0,xmm0

  if (fabsf(sq) < 0.000099999997) {
 vcvtss2sd   xmm5,xmm2,xmm2
 vcomisd     xmm5,mmword ptr [__real@3f1a36e2ddea900c (07FF76B192BE8h)]
 jae         xe::cpu::backend::x64::EmulateBungieNormalize3d+63h
(07FF76AEA6C33h) ctx->f[1] = sq;

}
 vxorpd      xmm0,xmm0,xmm0
 vmovsd      qword ptr [rdx+128h],xmm0
 add         rsp,18h
 ret
    ctx->f[1] = 0;
    return;
  }

  float inv = 1.0f / sq;
 vmovss      xmm0,dword ptr [__real@3f800000 (07FF76B168628h)]
 vdivss      xmm2,xmm0,xmm2

  x *= inv;
  y *= inv;
 vmulss      xmm0,xmm2,xmm3
 vmulss      xmm1,xmm2,xmm1
 vmovss      dword ptr [x],xmm1
  sto_be_f(vec, x);
 mov         eax,dword ptr [rsp+30h]
 movbe       dword ptr [r8],eax

  x *= inv;
  y *= inv;
 vmovss      dword ptr [rsp+30h],xmm0
  sto_be_f(&vec[1], y);
 mov         eax,dword ptr [rsp+30h]
 movbe       dword ptr [r8+4],eax
  z *= inv;
 vmulss      xmm0,xmm2,xmm4
 vmovss      dword ptr [rsp+30h],xmm0
  sto_be_f(&vec[2], z);
 mov         eax,dword ptr [rsp+30h]
 movbe       dword ptr [r8+8],eax
  ctx->f[1] = sq;

}
 vmovsd      qword ptr [rdx+128h],xmm5
  ctx->f[1] = sq;

}
 add         rsp,18h
 ret
*/
static double EmulateBungieNormalize3d(  // ppc::PPCContext_s* ctx,
    unsigned* vec) {
//  __debugbreak();
#if 0
  float x = load_be_f(vec);
  float y = load_be_f(&vec[1]);
  float z = load_be_f(&vec[2]);


  float sq = sqrtf((x * x) + (y * y) + (z * z));

  if (fabsf(sq) < 0.000099999997) {
    ctx->f[1] = 0;
    return;
  }

  float inv = 1.0f / sq;

  x *= inv;
  y *= inv;
  z *= inv;
  sto_be_f(vec, x);
  sto_be_f(&vec[1], y);
  sto_be_f(&vec[2], z);
  ctx->f[1] = sq;
#else
  __m128i xandy = _mm_loadu_si64(vec);
  __m128 z = _mm_load_ss((float*)&vec[2]);

  z = _mm_movelh_ps(_mm_castsi128_ps(xandy), z);
  __m128i mask =
      _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
  __m128i shufed = _mm_shuffle_epi8(_mm_castps_si128(z), mask);

  __m128 asfloats = _mm_castsi128_ps(shufed);

  __m128 squared = _mm_mul_ps(asfloats, asfloats);
  __m128 sum1 = _mm_hadd_ps(squared, squared);
  __m128 sum2 = _mm_hadd_ps(sum1, sum1);
  __m128 sqr = _mm_sqrt_ss(sum2);
  __m128 inv = _mm_rcp_ss(sqr);

  __m128 scaled_for_sto = _mm_mul_ps(_mm_broadcastss_ps(inv), asfloats);
  // ctx->f[1] = sqr.m128_f32[0];
  if (fabsf(sqr.m128_f32[0]) < 0.000099999997f) {
    return 0.0;
  }

  __m128i bswapped = _mm_shuffle_epi8(_mm_castps_si128(scaled_for_sto), mask);

  _mm_storeu_si64(vec, bswapped);

  vec[2] = bswapped.m128i_u32[2];
  return sqr.m128_f32[0];
#endif
}
static void memcpy_forwarder(void*, void* dst, void* src, unsigned size) {
  // __debugbreak();
  memcpy(dst, src, size);
}
static void memmove_forwarder(void*, void* dst, void* src, unsigned size) {
  // __debugbreak();
  memmove(dst, src, size);
}

#define PPCFIELD_PTR(field) \
  e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, field)]
#define PPCFIELD_PTR_SIZED(field, sz) \
  e.sz[e.GetContextReg() + offsetof(ppc::PPCContext_s, field)]

#define PPCFIELD_PTR_SIZED_WITH_OFFSET(field, sz, offset) \
  e.sz[e.GetContextReg() + offsetof(ppc::PPCContext_s, field) + offset]
static void emit_strncmp(builtin_emitter_t& ee) {
  /*
          int r = 0;

          while (n--
                     && ((r = ((int)(*((unsigned char *)s1))) - *((unsigned char
  *)s2++))
                          == 0)
                     && *s1++);

          return r;

          mov     edx, edx
          xor     ecx, ecx
  .L3:
          cmp     rdx, rcx
          je      .L4
          movzx   eax, BYTE PTR [rdi+rcx]
          movzx   r8d, BYTE PTR [rsi+rcx]
          mov     r9d, eax
          sub     eax, r8d
          jne     .L1
          add     rcx, 1
          test    r9b, r9b
          jne     .L3
          ret
  .L4:
          xor     eax, eax
  .L1:
          ret
  */

      ee.call_named_function("ppc_strncmp", [&ee](Xbyak::CodeGenerator& e) {
        e.mov(e.eax, PPCFIELD_PTR_BLTIN(r[3]));
        e.mov(e.edx, PPCFIELD_PTR_BLTIN(r[4]));
        // e.mov(e.ecx, PPCFIELD_PTR(r[5]));

        translate_address_in_register(e, e.rax);
        translate_address_in_register(e, e.rdx);
        e.xor_(e.ecx, e.ecx);

        Xbyak::Label L3, L4, L1;
        e.L(L3);
        e.cmp(e.ecx, PPCFIELD_PTR_BLTIN(r[5]));
        e.je(L4);
        e.movzx(e.ebp, e.ptr[e.rax + e.rcx]);
        e.movzx(e.r8d, e.ptr[e.rdx + e.rcx]);
        e.mov(e.r9d, e.ebp);
        e.sub(e.ebp, e.r8d);
        e.jne(L1);
        e.add(e.ecx, 1);
        e.test(e.r9b, e.r9b);
        e.jne(L3);
        e.jmp(L1);
        e.L(L4);
        e.xor_(e.ebp, e.ebp);
        e.L(L1);
        e.mov(PPCFIELD_PTR_BLTIN(r[3]), e.rbp);
        e.ret();
      });

}

static void do_slowfunc(X64Emitter& e) {
  // ptr
  //(uint8_t *param_1, uint32_t param_2)
  /*
    uint32_t uVar1;
uint32_t uVar2;
bool bVar3;
int64_t iVar4;
int32_t iVar5;
uint32_t* puVar6;
  */
  e.mov(e.eax, PPCFIELD_PTR(r[3]));
  e.mov(e.edx, PPCFIELD_PTR(r[4]));
  ;

  translate_address_in_register(e, e.rax);
  Xbyak::Label end;
  Xbyak::Label skip_first_branch;

  /*
      iVar5 = (uint32_t)param_1[1] + (uint32_t)*param_1;
      uVar1 = *(uint32_t*)(param_1 + 8);
  */
  e.movzx(e.ecx, e.byte[e.rax + 1]);
  e.movzx(e.ebp, e.byte[e.rax]);
  e.add(e.ecx, e.ebp);
  e.movbe(e.ebp, e.ptr[e.rax + 8]);

  /*
        if ((int32_t)uVar1 < iVar5) {
      uVar2 = *(uint32_t*)(param_1 + uVar1 * 0xc + 0x10);
      if ((param_2 < uVar2) ||
          (bVar3 = true,
           *(int32_t*)(param_1 + uVar1 * 0xc + 0x18) + uVar2 <= param_2)) {
        bVar3 = false;
      }
      if (bVar3) {
        return (uint64_t)uVar1;
      }
    }
  */

  e.cmp(e.ebp, e.ecx);
  e.jge(skip_first_branch);

  // uvar2 = r8

  e.mov(e.r8d, e.ebp);

#if 0




  iVar4 = 0;
  if (iVar5 != 0) {
    puVar6 = (uint32_t*)(param_1 + 0x10);
    do {
      if ((param_2 < *puVar6) ||
          (bVar3 = true, puVar6[2] + *puVar6 <= param_2)) {
        bVar3 = false;
      }
      if (bVar3) {
        *(int32_t*)(param_1 + 8) = (int32_t)iVar4;
        return iVar4;
      }
      iVar4 = iVar4 + 1;
      puVar6 = puVar6 + 3;
    } while ((int32_t)iVar4 < iVar5);
  }
  return 0xffffffffffffffff;
#endif
}
static inline uint64_t sub_824EAF38(uint32_t param_1_, uint32_t param_2,
                                    uintptr_t membase) {
  uint32_t uVar1;
  uint32_t uVar2;
  bool bVar3;
  int64_t iVar4;
  int32_t iVar5;
  uint32_t* puVar6;

  uint8_t* param_1 = (uint8_t*)(membase + param_1_);

  iVar5 = (uint32_t)param_1[1] + (uint32_t)*param_1;
  uVar1 = _load_be_u32((void*)(param_1 + 8));
  if ((int32_t)uVar1 < iVar5) {
    uVar2 = _load_be_u32(param_1 + uVar1 * 0xc + 0x10);
    if ((param_2 < uVar2) ||
        (bVar3 = true,
         _loadbe_i32(param_1 + uVar1 * 0xc + 0x18) + uVar2 <= param_2)) {
      bVar3 = false;
    }
    if (bVar3) {
      return (uint64_t)uVar1;
    }
  }
  iVar4 = 0;
  if (iVar5 != 0) {
    puVar6 = (uint32_t*)(param_1 + 0x10);
    do {
      if ((param_2 < _load_be_u32(puVar6)) ||
          (bVar3 = true,
           _load_be_u32(&puVar6[2]) + _load_be_u32(puVar6) <= param_2)) {
        bVar3 = false;
      }
      if (bVar3) {
        //*(int32_t*)(param_1 + 8) = (int32_t)iVar4;
        _storebe_i32(param_1 + 8, iVar4);
        return iVar4;
      }
      iVar4 = iVar4 + 1;
      puVar6 = puVar6 + 3;
    } while ((int32_t)iVar4 < iVar5);
  }
  return 0xffffffffffffffff;
}

static int64_t match_red_dead_redemption_freqcall_824EB0C8(uint32_t param_1_,
                                                           uint64_t param_2,
                                                           uintptr_t membase) {
  int32_t iVar2;
  int64_t iVar1;
  int32_t* in_r13;
  uint32_t* param_1 = (uint32_t*)(param_1_ + membase);

  iVar2 = sub_824EAF38((uint64_t)_load_be_u32(param_1), param_2, membase);
  if (iVar2 == -1) {
    iVar1 = 0;
  } else {
    iVar2 = iVar2 * 0xc + _load_be_u32(param_1);
    uint8_t* iVar2_ = (uint8_t*)(iVar2 + membase);

    iVar1 = (uint64_t)_load_be_u32(iVar2_ + 0x14) -
            (uint64_t) * (uint32_t*)(iVar2_ + 0x10);
  }
  return iVar1;
}
static void red_dead_redemption_freqcall_824EAF38_thunk(
    void*, ppc::PPCContext_s* ppc_ctx, uintptr_t membase) {
  ppc_ctx->r[3] = sub_824EAF38(ppc_ctx->r[3], ppc_ctx->r[4], membase);
}

static void red_dead_redemption_freqcall_824EB0C8_thunk(
    void*, ppc::PPCContext_s* ppc_ctx, uintptr_t membase) {
  ppc_ctx->r[3] = match_red_dead_redemption_freqcall_824EB0C8(
      ppc_ctx->r[3], ppc_ctx->r[4], membase);
}

static void memset_forwarder(void*, ppc::PPCContext_s* ctx, uint8_t* membase) {
  memset(((uint32_t)ctx->r[3]) + membase, ((uint32_t)ctx->r[4]),
         ((uint32_t)ctx->r[5]));
}

static void emit_wcslen(builtin_emitter_t& ee) {
  ee.call_named_function("ppc_wcslen", [&ee](Xbyak::CodeGenerator& e) {
    Xbyak::Label end, loopbody;

    e.mov(e.eax,
          e.ptr[ee->GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);

    e.xor_(e.ecx, e.ecx);
    e.add(e.rax, ee->GetMembaseReg());
    // translate_address_in_register(e, e.rax);
    // zero high bits of rdx and edx to break any deps
    e.xor_(e.edx, e.edx);
    e.L(loopbody);
    e.mov(e.dx, e.ptr[e.rax + e.rcx]);
    e.test(e.dx, e.dx);
    e.jz(end);
    e.add(e.ecx, 2);
    e.jmp(loopbody);
    e.L(end);
    e.shr(e.ecx, 1);
    e.mov(e.ptr[ee->GetContextReg() + offsetof(ppc::PPCContext_s, r[3])],
          e.rcx);
    e.ret();
  });
}

struct strcmpi_table_t {
  uint64_t arr[4];
};
// using strcmpi_table_t = std::array<uint64_t, 4>;

static constexpr strcmpi_table_t compute_strcmpi_table() {
  strcmpi_table_t result{};

  for (unsigned i = 0; i < 256; ++i) {
    if ((0x40 < i) && (i < 0x5b)) {
      result.arr[i / 64] |= (1ull << (i & 63));
    }
  }
  return result;
}
static constexpr strcmpi_table_t g_strcmpi_table = compute_strcmpi_table();

static void strcmpi_thunk(void*, ppc::PPCContext_s* ctx, char* membase) {
  ctx->r[3] =
      strcmpi(membase + (unsigned)ctx->r[3], membase + (unsigned)ctx->r[4]);
}

static void emit_strchr(builtin_emitter_t& ee) {
  /*
          movzx   edx, BYTE PTR [rdi]
      test    esi, esi
      jne     .L2


  */

  // edi = r8
  // esi = ebp


      ee.call_named_function("ppc_strchr", [&ee](Xbyak::CodeGenerator& e) {
        Xbyak::Label l4, l2, l6, l5, l8, l10, l7, end, endskipassign;
        e.mov(e.r8d, PPCFIELD_PTR_BLTIN(r[3]));
        e.mov(e.ebp, PPCFIELD_PTR_BLTIN(r[4]));
        // translate_address_in_register(e, e.r8);
        e.add(e.r8, ee->GetMembaseReg());
        e.movzx(e.edx, e.byte[e.r8]);
        e.test(e.ebp, e.ebp);
        e.mov(e.rcx, e.r8);
        e.jne(l2);

        e.mov(e.rax, e.r8);
        e.test(e.rdx, e.rdx);
        e.je(l7);
        e.L(l4);
        e.add(e.rax, 1);
        e.cmp(e.byte[e.rax], 0);
        e.jne(l4);
        e.jmp(end);

        e.L(l2);
        e.test(e.rdx, e.rdx);
        e.je(l8);
        e.cmp(e.ebp, e.edx);
        e.jne(l5);
        e.jmp(l10);

        e.L(l6);
        e.cmp(e.edx, e.ebp);
        e.je(l10);

        /*
            mov     rax, rdi
            test    rdx, rdx
            je      .L7
      .L4:
            add     rax, 1
            cmp     BYTE PTR [rax], 0
            jne     .L4
            ret
      .L2:
            test    rdx, rdx
            je      .L8
            cmp     esi, edx
            jne     .L5
            jmp     .L10
      .L6:
            cmp     edx, esi
            je      .L10

        */

        /*
        .L5:
            movzx   edx, BYTE PTR [rdi+1]
            add     rdi, 1
            test    dl, dl
            jne     .L6
      .L8:
            xor     eax, eax
            ret
      .L10:
            mov     rax, rdi
            ret
      .L7:
            ret
        */
        e.L(l5);
        e.movzx(e.edx, e.byte[e.r8 + 1]);
        e.add(e.r8, 1);
        e.test(e.dl, e.dl);
        e.jne(l6);

        e.L(l8);
        e.xor_(e.eax, e.eax);
        e.mov(PPCFIELD_PTR_BLTIN(r[3]), e.rax);

        e.jmp(endskipassign);

        e.L(l10);
        e.mov(e.rax, e.r8);
        e.jmp(end);

        e.L(l7);
        e.jmp(end);

        e.L(end);
        e.sub(e.rax, e.rcx);
        e.add(PPCFIELD_PTR_BLTIN(r[3]), e.eax);
        e.L(endskipassign);
        e.ret();
      });
}

static void emit_strcmpi(X64Emitter& e) {
#if 0
  e.mov(e.eax, PPCFIELD_PTR(r[3]));
  e.mov(e.edx, PPCFIELD_PTR(r[4]));
  e.push(e.rbp);
  e.mov(e.rbp, (uintptr_t)&g_strcmpi_table.arr[0]);
  e.dec(e.eax);
  e.dec(e.edx);
  e.xor_(e.r8d, e.r8d);
  e.xor_(e.r9d, e.r9d);
  e.xor_(e.ecx, e.ecx);

  translate_address_in_register(e, e.rax);
  translate_address_in_register(e, e.rdx);
  e.push(e.rbx);
  
  e.push(e.r9);
  e.push(e.r8);
  // i can use rax, rcx, rdx, r8, and r9 without saving them??

  /*
  lbzu      r6, 1(r4)
lbzu      r5, 1(r9)
cmpwi     cr7, r6, 0
subf.     r3, r6, r5
beq       cr7, locret_82BAEDA0
  */

  Xbyak::Label locret_82BAEDA0;
  Xbyak::Label loc_82BAED58;
  e.L(loc_82BAED58);
  e.add(e.rdx, 1);
  e.add(e.rax, 1);
  e.mov(e.r8b, e.ptr[e.rdx]);
  e.mov(e.r9b, e.ptr[e.rax]);

  e.mov(e.cl, e.r8b);
  e.sub(e.cl, e.r9b);
  e.test(e.r8b, e.r8b);
  e.jz(locret_82BAEDA0);
  e.test(e.cl, e.cl);
  e.jz(loc_82BAED58);


  //e.xor_(e.bl, e.bl);

  e.movzx(e.ebx, e.r8b);
  e.shr(e.ebx, 6); //div by 64
  e.mov(e.rbx, e.ptr[e.rbp+e.rbx*8]);
  e.bt(e.rbx, e.r8);

  e.setc(e.ch);
  e.movzx(e.ebx, e.r9b);
  e.shr(e.ebx, 6);
  e.shl(e.ch, 5);

  e.mov(e.rbx, e.ptr[e.rbp+e.rbx*8]);
  e.or_(e.r8b, e.ch);
  e.bt(e.rbx, e.r9);
  e.setc(e.ch);
  e.shl(e.ch, 5);
  e.or_(e.r9b, e.ch);
  e.mov(e.cl, e.r8b);
  e.sub(e.cl, e.r9b);
  e.jz(loc_82BAED58);
  e.L(locret_82BAEDA0);
  e.movsx(e.ecx, e.cl);
  e.movsxd(e.rcx, e.ecx);
  e.pop(e.r8);
  e.pop(e.r9);
  e.pop(e.rbx);
  e.pop(e.rbp);
  e.mov(PPCFIELD_PTR(r[3]), e.rcx);

#else

  e.mov(e.rdx, e.GetContextReg());
  e.mov(e.r8, e.GetMembaseReg());
  e.CallNativeSafe(&strcmpi_thunk);
#endif
  /*
          cmpwi     cr5, r6, 0x41 # 'A'
.text:82BAED74 2F 06 00 5A                 cmpwi     cr6, r6, 0x5A # 'Z'
.text:82BAED78 41 94 00 0C                 blt       cr5, loc_82BAED84
.text:82BAED7C 41 99 00 08                 bgt       cr6, loc_82BAED84
.text:82BAED80 60 C6 00 20                 ori       r6, r6, 0x20 # ' '
.text:82BAED84
.text:82BAED84             loc_82BAED84:
.text:82BAED84
.text:82BAED84 2C 05 00 41                 cmpwi     r5, 0x41 # 'A'
.text:82BAED88 2C 85 00 5A                 cmpwi     cr1, r5, 0x5A # 'Z'
.text:82BAED8C 41 80 00 0C                 blt       loc_82BAED98
.text:82BAED90 41 85 00 08                 bgt       cr1, loc_82BAED98
.text:82BAED94 60 A5 00 20                 ori       r5, r5, 0x20 # ' '
.text:82BAED98
.text:82BAED98             loc_82BAED98:
.text:82BAED98
.text:82BAED98 7C 66 28 51                 subf.     r3, r6, r5
.text:82BAED9C 41 82 FF BC                 beq       loc_82BAED58
  */

  //(0x40 < uVar3) && (uVar3 < 0x5b)
  // uVar3 > 0x40 & uVar3 < 0x5b
  // 0x40 < uVar3) && (uVar3 < 0x5b)
}

static void emit_d3dresource_get_type(X64Emitter& e) {
  Xbyak::Label L13, L14, L1, L5, L4, L6, end;

  /*
          mov     edx, DWORD PTR [rdi]
      mov     eax, edx
      and     eax, 15
      cmp     eax, 3
      je      .L13
      cmp     eax, 4
      je      .L14
.L1:
      ret
  */
  e.push(e.r8);
  e.mov(e.r8, PPCFIELD_PTR(r[3]));

  translate_address_in_register(e, e.r8);

  e.movbe(e.edx, e.ptr[e.r8]);

  e.mov(e.eax, e.edx);
  e.and_(e.eax, 15);
  e.cmp(e.eax, 3);

  e.je(L13);
  e.cmp(e.eax, 4);
  e.je(L14);
  e.L(L1);
  e.jmp(end);

  /*
  .L14:
      and     edx, 1073741824
      je      .L1
      mov     edx, DWORD PTR [rdi+24]
      mov     edx, DWORD PTR [edx+48]
      and     edx, 1536
      cmp     edx, 1024
      mov     edx, 16
      cmove   eax, edx
      ret
  */
  e.L(L14);
  e.and_(e.edx, 1073741824);
  e.je(L1);
  e.movbe(e.edx, e.ptr[e.r8 + 24]);
  translate_address_in_register(e, e.rdx);
  e.movbe(e.edx, e.ptr[e.rdx + 48]);
  e.and_(e.edx, 1536);
  e.cmp(e.edx, 1024);
  e.mov(e.edx, 16);
  e.cmove(e.eax, e.edx);
  e.jmp(end);
  /*
    .L13:
          mov     edx, DWORD PTR [rdi+48]
          shr     edx, 9
          and     edx, 3
          cmp     edx, 3
          je      .L4
          cmp     edx, 2
          je      .L5
          test    edx, edx
          je      .L6
          test    BYTE PTR [rdi+33], 4
          mov     edx, 19
          cmovne  eax, edx
          ret
  */
  e.L(L13);
  e.movbe(e.edx, e.ptr[e.r8 + 48]);
  e.shr(e.edx, 9);
  e.and_(e.edx, 3);
  e.cmp(e.edx, 3);
  e.je(L4);
  e.cmp(e.edx, 2);
  e.je(L5);
  e.test(e.edx, e.edx);
  e.je(L6);
  e.test(e.byte[e.r8 + 33], 4);
  e.mov(e.edx, 19);
  e.cmovne(e.eax, e.edx);
  e.jmp(end);
  e.L(L5);
  e.mov(e.eax, 17);
  e.jmp(end);
  e.L(L4);
  e.mov(e.eax, 18);
  e.jmp(end);
  e.L(L6);
  e.mov(e.eax, 20);

  e.L(end);
  e.mov(PPCFIELD_PTR(r[3]), e.rax);
  e.pop(e.r8);
  e.DebugBreak();
  /*



  .L5:
          mov     eax, 17
          ret
  .L4:
          mov     eax, 18
          ret
  .L6:
          mov     eax, 20
          ret
  */
}
static unsigned ncalls_profile = 0;
static unsigned r13ptr = 0;

/*
    thanks to bmi we cut this down from like 100 emitted instructions to like 15
*/
static void emit_allocator_subfn(X64Emitter& e) {
  /*
     uint32_t uVar1;
 uint32_t uVar2;

 uVar1 = param_1[2];
 if (uVar1 == 0) {
    return 0;
 }
 uVar2 = -uVar1 & uVar1;
 param_1[2] = uVar1 ^ uVar2;
 return (int64_t)(int32_t)(((((uint32_t)((uVar2 & 0xffff0000) != 0) * 2 |
(uint32_t)((uVar2 & 0xff00ff00) != 0)) << 1 | (uint32_t)((uVar2 & 0xf0f0f0f0) !=
0)) << 1 | (uint32_t)((uVar2 & 0xcccccccc) != 0)) << 1 | (uint32_t)((uVar2 &
0xaaaaaaaa) != 0)) * (int64_t)(int32_t)param_1[1] + (uint64_t)*param_1;
}
  */
  // actually tzcnt
  // e.DebugBreak();
  e.mov(e.eax, PPCFIELD_PTR(r[3]));
  translate_address_in_register(e, e.rax);
  e.movbe(e.r8d, e.ptr[e.rax]);
  e.movbe(e.ebp, e.ptr[e.rax + 4]);
  e.movbe(e.ecx, e.ptr[e.rax + 8]);

  Xbyak::Label badboi;

  e.test(e.ecx, e.ecx);
  e.jz(badboi);

  /* e.mov(e.edx, e.ecx);
   e.neg(e.edx);
   e.and_(e.edx, e.ecx);
   */
  e.blsr(e.edx, e.ecx);
  e.movbe(e.ptr[e.rax + 8], e.edx);
  e.tzcnt(e.ecx, e.ecx);
  e.imul(e.ecx, e.ebp);
  e.add(e.ecx, e.r8d);
  e.L(badboi);
  e.mov(PPCFIELD_PTR(r[3]), e.ecx);
}
unsigned ncallsipow = 0;

unsigned char byte_82075D70[] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
    0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23,
    0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,
    0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
    0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73,
    0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x5B, 0x2F, 0x5D, 0x5E, 0x5F,
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B,
    0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
    0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B,
    0x9C, 0x9D, 0x9E, 0x9F, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3,
    0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF,
    0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB,
    0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
    0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0, 0xE1, 0xE2, 0xE3,
    0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB,
    0xFC, 0xFD, 0xFE, 0xFF};

/*
    located the same function in Max Payne 3 for the pc (same engine).
    this is its decompilation, it seems to be equivalent
*/
unsigned int __cdecl sub_822FD530(unsigned __int8* a1, unsigned int a2) {
  unsigned __int8* v2;  // ecx
  unsigned int result;  // eax
  unsigned __int8 v4;   // dl
  unsigned __int8* v5;  // ecx
  unsigned int v6;      // eax
  unsigned int v7;      // eax
  int v8;               // edx
  unsigned char* v9;    // ecx
  int v10;              // esi
  int v11;              // eax
  unsigned int v12;     // edx
  int v13;              // eax
  unsigned __int8 i;    // dl
  unsigned int v15;     // eax

  v2 = a1;
  // if (!a1) return 0;
  result = a2;
  if (*a1 == 34) {
    v4 = a1[1];
    v5 = a1 + 1;
    if (v4) {
      do {
        if (v4 == 34) break;
        v6 = 1025 * ((unsigned __int8)byte_82075D70[v4] + result);
        ++v5;
        result = (v6 >> 6) ^ v6;
        v4 = *v5;
      } while (*v5);
    }
  } else {
    if (!((*(unsigned*)a1 - (unsigned)0x1010101) & 0x80808080)) {
      do {
        v7 = 1025 * ((unsigned __int8)byte_82075D70[*v2] + result);
        v8 = (unsigned __int8)byte_82075D70[v2[1]];
        v9 = (v2 + 1);
        v10 = v7 ^ (v7 >> 6);
        v11 = (unsigned __int8)byte_82075D70[*(unsigned __int8*)(v9++ + 1)];
        v12 = 1025 *
              ((1025 * (v10 + v8) ^ ((unsigned int)(1025 * (v10 + v8)) >> 6)) +
               v11);
        v13 = (unsigned __int8)byte_82075D70[*(unsigned __int8*)(v9 + 1)];
        v2 = (unsigned __int8*)(v9 + 2);
        result = (1025 * ((v12 ^ (v12 >> 6)) + v13) >> 6) ^
                 1025 * ((v12 ^ (v12 >> 6)) + v13);
      } while (!((*(unsigned*)v2 - 0x1010101) & 0x80808080));
    }
    for (i = *v2; *v2; i = *v2) {
      v15 = 1025 * ((unsigned __int8)byte_82075D70[i] + result);
      ++v2;
      result = (v15 >> 6) ^ v15;
    }
  }
  return result;
}

static bool emit_memcpy_const_size(X64Emitter& e, const Instr* instruction,
                                   unsigned size) {
  // return false;
  if (size > 256) return false;
  unsigned i = 0;

  bool did_zero_ymm = false;
  e.mov(e.edx, PPCFIELD_PTR(r[3]));
  e.mov(e.eax, PPCFIELD_PTR(r[4]));

  translate_address_in_register(e, e.rdx);
  translate_address_in_register(e, e.rax);
  unsigned nbytes_left = size;

  while (nbytes_left >= 128) {
    e.vmovdqu(e.ymm0, e.ptr[e.rax + i]);
    e.vmovdqu(e.ymm1, e.ptr[e.rax + i + 32]);
    e.vmovdqu(e.ymm2, e.ptr[e.rax + i + 64]);
    e.vmovdqu(e.ptr[e.rdx + i], e.ymm0);
    e.vmovdqu(e.ymm3, e.ptr[e.rax + i + 96]);
    e.vmovdqu(e.ptr[e.rdx + i + 32], e.ymm1);
    e.vmovdqu(e.ptr[e.rdx + i + 64], e.ymm2);
    e.vmovdqu(e.ptr[e.rdx + i + 96], e.ymm3);
    nbytes_left -= 128;
    i += 128;
  }
  while (nbytes_left >= 64) {
    e.vmovdqu(e.ymm0, e.ptr[e.rax + i]);
    e.vmovdqu(e.ymm1, e.ptr[e.rax + i + 32]);

    e.vmovdqu(e.ptr[e.rdx + i], e.ymm0);
    e.vmovdqu(e.ptr[e.rdx + i + 32], e.ymm1);
    i += 64;
    nbytes_left -= 64;
  }

  while (nbytes_left >= 32) {
    e.vmovdqu(e.ymm0, e.ptr[e.rax + i]);
    e.vmovdqu(e.ptr[e.rdx + i], e.ymm0);
    i += 32;
    nbytes_left -= 32;
  }

  while (nbytes_left >= 16) {
    e.vmovdqu(e.xmm0, e.ptr[e.rax + i]);
    e.vmovdqu(e.ptr[e.rdx + i], e.xmm0);
    i += 16;
    nbytes_left -= 16;
  }

  while (nbytes_left >= 8) {
    e.mov(e.rcx, e.ptr[e.rax + i]);

    e.mov(e.ptr[e.rdx + i], e.rcx);
    i += 8;
    nbytes_left -= 8;
  }

  while (nbytes_left >= 4) {
    e.mov(e.ecx, e.ptr[e.rax + i]);

    e.mov(e.ptr[e.rdx + i], e.ecx);
    i += 4;
    nbytes_left -= 4;
  }
  while (nbytes_left >= 2) {
    e.mov(e.cx, e.ptr[e.rax + i]);

    e.mov(e.ptr[e.rdx + i], e.cx);
    i += 2;
    nbytes_left -= 2;
  }
  while (nbytes_left >= 1) {
    e.mov(e.cl, e.ptr[e.rax + i]);

    e.mov(e.ptr[e.rdx + i], e.cl);
    i += 1;
    nbytes_left -= 1;
  }
  return false;
}
static bool emit_memzero_const_size(X64Emitter& e, const Instr* instruction,
                                    unsigned size, unsigned val) {
  // return false;
  if (val != 0) return false;

  if (size > 256) return false;
  unsigned i = 0;

  bool did_zero_ymm = false;

  e.mov(e.edx, PPCFIELD_PTR(r[3]));

  // if(val == 0) {

  if ((size % 16)) e.xor_(e.eax, e.eax);

  if (size >= 32)
    e.vxorpd(e.ymm0, e.ymm0);
  else if (size >= 16)
    e.vxorpd(e.xmm0, e.xmm0);

#if 0
  }
  else {
    unsigned char bs[8];
    for(unsigned i = 0; i < 8; ++i)
        bs[i]=val;



    e.mov(e.rax, *reinterpret_cast<uint64_t*>(&bs[0]));
    if(size >= 16) {
        //reuse the context stored byte for the memset arg to set ymm0
        e.vpbroadcastb(e.ymm0, PPCFIELD_PTR_SIZED(r[4], byte));

        //e.vpinsrb(e.ymm0, e.al, 0);
        //e.vmovq(e.xmm0, e.rax);

        //e.vpbroadcastd(e.ymm0, PPCFIELD_PTR(r[4]));
        /*if(size>=32){
            

        }*/
    }


  }
#endif
  translate_address_in_register(e, e.rdx);

  unsigned nbytes_left = size;

  while (nbytes_left >= 32) {
    e.vmovupd(e.ptr[e.rdx + i], e.ymm0);

    nbytes_left -= 32;
    i += 32;
  }

  while (nbytes_left >= 16) {
    e.vmovupd(e.ptr[e.rdx + i], e.xmm0);
    nbytes_left -= 16;
    i += 16;
  }

  while (nbytes_left >= 8) {
    e.mov(e.ptr[e.rdx + i], e.rax);
    nbytes_left -= 8;
    i += 8;
  }

  while (nbytes_left >= 4) {
    e.mov(e.ptr[e.rdx + i], e.eax);
    nbytes_left -= 4;
    i += 4;
  }
  while (nbytes_left >= 2) {
    e.mov(e.ptr[e.rdx + i], e.ax);
    nbytes_left -= 2;
    i += 2;
  }

  while (nbytes_left >= 1) {
    e.mov(e.ptr[e.rdx + i], e.al);
    nbytes_left -= 1;
    i += 1;
  }

  return true;
}
std::map<unsigned, uint64_t> m_calls_to_loc;

bool installed_atexit_handler = false;
std::mutex callmutex;
static void doatexit() {
  FILE* outfile = fopen("callcounts.txt", "w");

  std::vector<std::pair<unsigned, unsigned>> sorted_counts;

  for (auto&& iter : m_calls_to_loc) {
    sorted_counts.push_back(iter);
    //
  }
  std::sort(sorted_counts.begin(), sorted_counts.end(),
            [](auto bleh, auto blah) { return bleh.second < blah.second; });

  for (auto&& iter : sorted_counts) {
    fprintf(outfile, "%X : %lld\n", iter.first, iter.second);
  }
  fclose(outfile);
}

static bool try_do_saveregs_emit(X64Emitter& e, const Instr* i,
                                 GuestFunction* gf) {
  using ppc::PPCBuiltin;
  bool have_r1_in_rbp = false;

  auto load_r1 = [&have_r1_in_rbp, &e]() {
    if (!have_r1_in_rbp) {
      e.mov(e.ebp, PPCFIELD_PTR(r[1]));
      translate_address_in_register(e, e.rbp);
      have_r1_in_rbp = true;
      return e.rbp;
    } else {
      return e.rbp;
    }
  };

  // if (i->flags & CALL_TAIL) return false;

//#define MOV_SAVED_REG     mov
#define MOV_SAVED_REG movbe
  auto save_reg = [&e, &load_r1](unsigned reg_to_save, unsigned negoffs,
                                 bool save32 = false) {
    auto r1 = load_r1();

    if (save32) {
      e.mov(e.edx, e.ptr[offsetof(ppc::PPCContext_s, r[0]) + (8 * reg_to_save) +
                         e.GetContextReg()]);

      e.MOV_SAVED_REG(e.ptr[r1 - negoffs], e.edx);
    } else {
      e.mov(e.rdx, e.ptr[offsetof(ppc::PPCContext_s, r[0]) + (8 * reg_to_save) +
                         e.GetContextReg()]);

      e.MOV_SAVED_REG(e.ptr[r1 - negoffs], e.rdx);
    }
  };

  auto rest_reg = [&e, &load_r1](unsigned reg_to_save, unsigned negoffs,
                                 bool save32 = false) {
    auto r1 = load_r1();

    if (save32) {
      e.MOV_SAVED_REG(e.edx, e.ptr[r1 - negoffs]);
      e.mov(e.ptr[offsetof(ppc::PPCContext_s, r[0]) + (8 * reg_to_save) +
                  e.GetContextReg()],
            e.rdx);

    } else {
      /*  e.mov(e.rax, e.ptr[offsetof(ppc::PPCContext_s, r[0]) + (8 *
         reg_to_save) + e.GetContextReg()]);*/

      e.MOV_SAVED_REG(e.rdx, e.ptr[r1 - negoffs]);
      e.mov(e.ptr[offsetof(ppc::PPCContext_s, r[0]) + (8 * reg_to_save) +
                  e.GetContextReg()],
            e.rdx);
    }
  };

#define expandcases_gpsaverestore(prefix) \
  case PPCBuiltin::prefix##gplr:          \
    prefix##_reg(14, 152);                \
  case PPCBuiltin::prefix##gplr15:        \
    prefix##_reg(15, 144);                \
  case PPCBuiltin::prefix##gplr16:        \
    prefix##_reg(16, 136);                \
  case PPCBuiltin::prefix##gplr17:        \
    prefix##_reg(17, 128);                \
  case PPCBuiltin::prefix##gplr18:        \
    prefix##_reg(18, 120);                \
  case PPCBuiltin::prefix##gplr19:        \
    prefix##_reg(19, 112);                \
  case PPCBuiltin::prefix##gplr20:        \
    prefix##_reg(20, 104);                \
  case PPCBuiltin::prefix##gplr21:        \
    prefix##_reg(21, 96);                 \
  case PPCBuiltin::prefix##gplr22:        \
    prefix##_reg(22, 88);                 \
  case PPCBuiltin::prefix##gplr23:        \
    prefix##_reg(23, 80);                 \
  case PPCBuiltin::prefix##gplr24:        \
    prefix##_reg(24, 72);                 \
  case PPCBuiltin::prefix##gplr25:        \
    prefix##_reg(25, 64);                 \
  case PPCBuiltin::prefix##gplr26:        \
    prefix##_reg(26, 56);                 \
  case PPCBuiltin::prefix##gplr27:        \
    prefix##_reg(27, 48);                 \
  case PPCBuiltin::prefix##gplr28:        \
    prefix##_reg(28, 40);                 \
  case PPCBuiltin::prefix##gplr29:        \
    prefix##_reg(29, 32);                 \
  case PPCBuiltin::prefix##gplr30:        \
    prefix##_reg(30, 24);                 \
  case PPCBuiltin::prefix##gplr31:        \
    prefix##_reg(31, 16);                 \
    prefix##_reg(12, 8, true)

  if (gf->ppc_builtin() >= PPCBuiltin::savegplr &&
      gf->ppc_builtin() <= PPCBuiltin::savegplr31 && (i->flags & CALL_TAIL)) {
    return false;
  }
  switch (gf->ppc_builtin()) {
    expandcases_gpsaverestore(save);
    break;
    
   expandcases_gpsaverestore(rest);
    e.mov(PPCFIELD_PTR(lr), e.rdx);

    if (i->flags & CALL_TAIL) {
      // If this is the last instruction in the last block, just let us
      // fall through.
      if (i->next || i->block->next)
        e.jmp(e.epilog_label(), CodeGenerator::T_AUTO);


    }
    break;
    
    default:
      return false;
  }
  return true;
}
static void emit_setpixelshader_constants(X64Emitter& e, bool vertshader) {
  /*
  addi      r10, r4, 0x178
  mr        r11, r5
  slwi      r10, r10, 4
  mr        r9, r6
  add       r10, r10, r3
  li        r8, 0x80
  li        r5, 0x10
  cmplwi    cr6, r6, 3
  ble       cr6, loc_822F0330
  */
  Xbyak::Label loc_822F0330, loc_822F02D4;

  Xbyak::Label loopbody;
  auto r3 = e.edx;
  auto r4 = e.ecx;
  auto r5 = e.eax;
  auto r6 = e.r8d;

  e.mov(e.edx, PPCFIELD_PTR(r[3]));
  e.mov(e.ecx, PPCFIELD_PTR(r[4]));
  e.mov(e.eax, PPCFIELD_PTR(r[5]));
  e.mov(e.r8d, PPCFIELD_PTR(r[6]));

  // move to (r4+0x178)*16 + r3 from r11
  e.add(e.ecx, vertshader ? 0x78 : 0x178);

  e.shl(e.ecx, 4);
  translate_address_in_register(e, e.rax);
  e.add(e.ecx, e.edx);
  // just bswap the mask, instead of doing movbe, or, movbe
  e.movbe(e.rbp, PPCFIELD_PTR(r[7]));
  translate_address_in_register(e, e.rdx);

  e.or_(e.ptr[e.rdx + (vertshader ? 0 : 8)], e.rbp);
  translate_address_in_register(e, e.rcx);
  e.xor_(e.edx, e.edx);
  /*
      mov 16 bytes from [rax] to [rcx] while r8d
  */

  e.L(loopbody);
  e.sub(e.r8d, 1);
  e.vmovupd(e.xmm0, e.ptr[e.rax + e.rdx]);

  e.vmovupd(e.ptr[e.rcx + e.rdx], e.xmm0);
  e.add(e.edx, 16);
  e.test(e.r8d, e.r8d);
  e.jnz(loopbody);

  /*
  ld        r11, 8(r3)
or        r11, r7, r11
std       r11, 8(r3)
  */
}

static void emit_red_dead_redemption_sub_82EB1110(X64Emitter& e) {
  /*
  int32_t * sub_82EB1110(int32_t *param_1, int32_t *param_2)
  {
     int32_t *piVar1;
     uint32_t uVar2;

     uVar2 = *param_2 * 0x1001;
     uVar2 = (uVar2 >> 0x16 ^ uVar2) * 0x11;
     uVar2 = (uVar2 >> 9 ^ uVar2) * 0x401;
     uVar2 = (uVar2 >> 2 ^ uVar2) * 0x81;
     uVar2 = uVar2 >> 0xc ^ uVar2;

  }
  */
  e.mov(e.eax, PPCFIELD_PTR(r[3]));
  e.mov(e.ecx, PPCFIELD_PTR(r[4]));
  translate_address_in_register(e, e.rax);
  translate_address_in_register(e, e.rcx);

  e.movbe(e.edx, e.ptr[e.rcx]);
  /*
      todo: ive seen this sequence before, im pretty sure it maps to parity or
     something, so this could be cut down further
  */

  e.imul(e.edx, e.edx, 0x1001);

  e.mov(e.ebp, e.edx);
  e.shr(e.ebp, 0x16);
  e.xor_(e.edx, e.ebp);
  e.movbe(e.r8d, e.ptr[e.rax]);

  e.imul(e.edx, e.edx, 0x11);

  e.xor_(e.r9d, e.r9d);

  e.mov(e.ebp, e.edx);
  e.shr(e.ebp, 9);
  e.xor_(e.edx, e.ebp);
  e.imul(e.edx, e.edx, 0x401);
  e.movbe(e.eax, e.ptr[e.rax + 8]);
  e.mov(e.ebp, e.edx);
  e.shr(e.ebp, 2);
  e.xor_(e.edx, e.ebp);
  e.imul(e.edx, e.edx, 0x81);
  e.mov(e.ebp, e.edx);
  e.shr(e.ebp, 0xc);
  e.xor_(e.edx, e.ebp);
  /*
piVar1 = *(int32_t**)(((uVar2 == 0) + uVar2 & *param_1 - 1U) * 8 + param_1[2]);
*/

  e.sete(e.r9b);
  e.dec(e.r8d);
  e.add(e.edx, e.r9d);
  e.and_(e.edx, e.r8d);
  e.lea(e.eax, e.ptr[e.eax + e.edx * 8]);
  translate_address_in_register(e, e.rax);
  e.movbe(e.edx, e.ptr[e.rax]);

  Xbyak::Label looper, end, retnull;

  // preload *param_2
  // normal load, since we're only comparing for equality byte order doesnt
  // matter
  e.mov(e.ecx, e.ptr[e.rcx]);
  e.L(looper);
  e.mov(e.eax, e.edx);
  e.test(e.edx, e.edx);
  e.jz(retnull);

  translate_address_in_register(e, e.rdx);
  // no big endian conv needed
  e.cmp(e.ptr[e.rdx], e.ecx);
  e.jz(end);
  e.movbe(e.edx, e.ptr[e.rdx + 8]);
  e.jmp(looper);
  e.L(end);
  e.add(e.eax, 4);
  e.L(retnull);
  e.mov(PPCFIELD_PTR(r[3]), e.eax);

  /*

while (true) {
if (piVar1 == (int32_t*)0x0) {
   return (int32_t*)0;
}
if (*param_2 == *piVar1) {
   break;
}
piVar1 = (int32_t*)piVar1[2];
}
return piVar1 + 1;
  */
}
// assume they already set up the params
static uint32_t emit_named_thunk(builtin_emitter_t& ee, const char* name,
                                 void* func) {
  /*
      dont touch this, the offsets and alignment is hand calculated
  */
  size_t thunk =
      ee.new_named_function(name, [&ee, func](Xbyak::CodeGenerator& e) {
        auto ctxreg = ee->GetContextReg();
        // R3–R10 = volatile
        // VR1–VR13 = volatile
        // R10, R11, XMM0-5

        e.mov(PPCFIELD_PTR_BLTIN(r[4]), e.r10);
        e.mov(PPCFIELD_PTR_BLTIN(r[5]), e.r11);
        e.vmovapd(PPCFIELD_PTR_BLTIN(v[1]), e.xmm4);
        e.vmovapd(PPCFIELD_PTR_BLTIN(v[2]), e.xmm5);

        e.call(e.ptr[e.rip + 26]);

        e.mov(e.r10, PPCFIELD_PTR_BLTIN(r[4]));
        e.mov(e.r11, PPCFIELD_PTR_BLTIN(r[5]));
        e.vmovapd(e.xmm4, PPCFIELD_PTR_BLTIN(v[1]));
        e.vmovapd(e.xmm5, PPCFIELD_PTR_BLTIN(v[2]));
        e.ret();
        e.db(0xcc);  // align to 8 bytes
        e.dq((uint64_t)func);
      });
  return thunk;
}

static void call_named_thunk(builtin_emitter_t& ee, const char* name,
                             void* func) {
  size_t thunk =
      ee.new_named_function(name, [&ee, func](Xbyak::CodeGenerator& e) {
        auto ctxreg = ee->GetContextReg();
        // R3–R10 = volatile
        // VR1–VR13 = volatile
        // R10, R11, XMM0-5

        e.mov(PPCFIELD_PTR_BLTIN(r[4]), e.r10);
        e.mov(PPCFIELD_PTR_BLTIN(r[5]), e.r11);
        e.vmovapd(PPCFIELD_PTR_BLTIN(v[1]), e.xmm4);
        e.vmovapd(PPCFIELD_PTR_BLTIN(v[2]), e.xmm5);
        e.mov(e.rax, (uint64_t)func);
        e.call(e.rax);
        // e.call(e.ptr[e.rip + 26]);

        e.mov(e.r10, PPCFIELD_PTR_BLTIN(r[4]));
        e.mov(e.r11, PPCFIELD_PTR_BLTIN(r[5]));
        e.vmovapd(e.xmm4, PPCFIELD_PTR_BLTIN(v[1]));
        e.vmovapd(e.xmm5, PPCFIELD_PTR_BLTIN(v[2]));
        e.ret();
        // e.db(0xcc);  // align to 8 bytes
        //  e.dq((uint64_t)func);
      });
  ee->mov(ee->rax, thunk);
  ee->call(ee->rax);
  // return thunk;
}
static void match_red_dead_redemption_switch_func_823C59F8_impl(
    uint32_t* param_1, unsigned param_2, uint32_t param_3) {
  unsigned parm1val = _load_be_u32(param_1);
  unsigned val0;
  switch (param_2) {
    case 0:
      val0 = param_3 << 0x1d | parm1val & 0x1fffffff;
      goto store_val0;
    case 1:
      val0 = (param_3 & 7) << 0x1a | parm1val & 0xe3ffffff;
      goto store_val0;
    case 2:
      val0 = (param_3 & 7) << 0x17 | parm1val & 0xfc7fffff;
      goto store_val0;
    case 3:
      // param_1[1] = param_3;
      _store_be_u32(&param_1[1], param_3);
      return;
    case 4:
      val0 = (param_3 & 7) << 0x14 | parm1val & 0xff8fffff;
      goto store_val0;
    case 5:
      val0 = (param_3 & 7) << 0x11 | parm1val & 0xfff1ffff;
      goto store_val0;
    case 6:
      val0 = (param_3 & 7) << 0xe | parm1val & 0xfffe3fff;
      goto store_val0;
    case 7:
      _store_be_u32(&param_1[2], param_3);
      return;
    case 8:
      val0 = (param_3 & 0xf) << 10 | parm1val & 0xffffc3ff;
      goto store_val0;
    case 9:
      val0 = (param_3 - 1) * 4 & 0x3c | parm1val & 0xffffffc3;
      goto store_val0;
    case 10:
      val0 = param_3 & 3 | parm1val & 0xfffffffc;
      goto store_val0;
    case 0xb:
      val0 = (param_3 & 0xf) << 6 | parm1val & 0xfffffc3f;
      goto store_val0;
    default:
      return;
  }

store_val0:
  _store_be_u32(param_1, val0);
}

static void do_guest_call_or_builtin(X64Emitter& e, const Instr* i,
                                     GuestFunction* gf) {
  using ppc::PPCBuiltin;

  builtin_emitter_t be{e, i};
  //
//  if (e.currentfunc == 0x8219a840) __debugbreak();
/*
  if(gf->address() == 0x822C92A8) {
    e.DebugBreak();
    e.Call(i, gf);
    return;
  }*/
#if 0
  e.Call(i, gf);
  return;
#endif
  /*
    0x822FD530 is called an insane number of times
  */

  /* if (gf->address() == 0x822F0290) {
     e.DebugBreak();
   }*/
  /* if (gf->address() == 0x82855A60 ) {
     e.mov(e.rax, (uintptr_t)&ncallsipow);
     e.inc(e.dword[e.rax]);
       /*e.mov(e.eax, PPCFIELD_PTR(r[3]));
       e.DebugBreak();
       e.Call(i, gf);*/
  //}*

  if (gf->ppc_builtin() == PPCBuiltin::Unclassified &&
      gf->address() > 0x80000000) {
    auto mem = e.backend()->processor()->memory();

    PPCBuiltin builtin = xe::cpu::ppc::classify_function_at(
        mem->TranslateVirtual(gf->address()), mem->physical_membase(),
        gf->module());

    gf->set_ppc_builtin(builtin);
  }

  if (gf->ppc_builtin() == PPCBuiltin::None) {
#if 0
    callmutex.lock();

    auto bleh = m_calls_to_loc.try_emplace(gf->address(), 0);
    e.mov(e.rax, (uintptr_t)&bleh.first->second);
    e.inc(e.qword[e.rax]);
    if (!installed_atexit_handler) {
      atexit(doatexit);
      installed_atexit_handler = true;
    }
    callmutex.unlock();
#endif
  }

  if (i->flags & CALL_TAIL) {
    e.Call(i, gf);
    return;
  }
  if (try_do_saveregs_emit(e, i, gf)) {
    return;
  }

  auto gfid = gf->ppc_builtin();
  if (false && gfid == ppc::PPCBuiltin::Memset) {
    const Value* set_to =
        hunt_context_store_backwards(i, offsetof(ppc::PPCContext_s, r[4]));
    const Value* namnt =
        hunt_context_store_backwards(i, offsetof(ppc::PPCContext_s, r[5]));

    if (set_to && namnt && set_to->IsConstant() && namnt->IsConstant() &&
        emit_memzero_const_size(e, i, namnt->constant.u32,
                                set_to->constant.u8)) {
    } else {
      /*e.mov(e.rdx, e.GetContextReg());
      e.mov(e.r8, e.GetMembaseReg());
      e.CallNativeSafe(memset_forwarder);*/
      be.load_gpreg_arg(3, INT32_TYPE, e.ecx);

      // e.mov(e.ecx, PPCFIELD_PTR(r[3]));
      be.load_gpreg_arg(4, INT32_TYPE, e.edx);
      be.load_gpreg_arg(5, INT32_TYPE, e.r8d);
      // e.mov(e.edx, PPCFIELD_PTR(r[4]));
      // e.mov(e.r8d, PPCFIELD_PTR(r[5]));
      translate_address_in_register(e, e.rcx);

      call_named_thunk(be, "ppc_memset",
                       GetProcAddress(GetModuleHandleA("ntdll.dll"), "memset"));
    }

  } else if (true && gfid == PPCBuiltin::wcslen_variant1) {
    emit_wcslen(be);
  } else if (true && gf->ppc_builtin() == PPCBuiltin::strchr_variant1) {
    emit_strchr(be);
  } else if (gf->ppc_builtin() ==
             PPCBuiltin::red_dead_redemption_sub_82EB1110) {
    emit_red_dead_redemption_sub_82EB1110(e);
  }
#if 1
  else if (gf->ppc_builtin() == PPCBuiltin::freqcall_c_allocator_) {
    emit_allocator_subfn(e);
  } else if (gf->ppc_builtin() ==
             PPCBuiltin::red_dead_redemption_tinyfunc_calledbillions) {
    /*
                lwz       r11, 0x100(r13)
.text:82C39264 80 6B 01 4C                 lwz       r3, 0x14C(r11)
.text:82C39268 4E 80 00 20                 blr
    */
    e.mov(e.eax, PPCFIELD_PTR(r[13]));
    translate_address_in_register(e, e.rax);
    e.movbe(e.eax, e.ptr[e.rax + 0x100]);
    translate_address_in_register(e, e.rax);
    e.movbe(e.eax, e.ptr[e.rax + 0x14c]);
    e.mov(PPCFIELD_PTR(r[3]), e.eax);
  } else if (gf->ppc_builtin() ==
             PPCBuiltin::d3d_device_set_pix_shader_constant_fn) {
    emit_setpixelshader_constants(e, false);

  } else if (gf->ppc_builtin() ==
             PPCBuiltin::d3d_device_set_vert_shader_constant_fn) {
    emit_setpixelshader_constants(e, true);

  } else if (gf->ppc_builtin() == PPCBuiltin::d3d_blocker_check) {
    e.mov(PPCFIELD_PTR_SIZED(r[3], dword), 1);
  }
#endif
  else if (true && gf->ppc_builtin() == PPCBuiltin::rdr_hashcode_function) {
    Xbyak::Label isnull;
    e.mov(e.ecx, PPCFIELD_PTR(r[3]));
    e.mov(e.edx, PPCFIELD_PTR(r[4]));
    e.mov(e.eax, e.ecx);
    e.test(e.ecx, e.ecx);
    e.jz(isnull);
    translate_address_in_register(e, e.rcx);
    call_named_thunk(be, "rdr_hashcode_fn_impl", sub_822FD530);

    e.L(isnull);
    e.mov(PPCFIELD_PTR(r[3]), e.eax);
    return;
  } else if (gf->ppc_builtin() == PPCBuiltin::flush_cached_mem_d3dlib) {
    // do nothing whatsoever!
  } else if (gf->ppc_builtin() == ppc::PPCBuiltin::Floorf1) {
    e.vroundsd(e.xmm0, PPCFIELD_PTR(f[1]), 9);
    e.vmovsd(PPCFIELD_PTR(f[1]), e.xmm0);

  } else if (gf->ppc_builtin() == PPCBuiltin::strcmpi_variant1) {
    emit_strcmpi(e);
  } else if (gf->ppc_builtin() == PPCBuiltin::get_exp_variant1) {
    e.mov(e.eax, PPCFIELD_PTR_SIZED_WITH_OFFSET(f[1], dword, 6));
    e.shr(e.eax, 4);
    e.and_(e.eax, 0x7FF);
    e.sub(e.eax, 0x3FE);
    e.mov(PPCFIELD_PTR(r[3]), e.eax);

  } else if (gf->ppc_builtin() == PPCBuiltin::set_exp_variant1) {
    /*

  v2 = a1;
  HIWORD(v2) = HIWORD(a1) & 0x800F | 16 * (a2 + 1022);
  return v2;
    */


    e.mov(e.eax, PPCFIELD_PTR_SIZED_WITH_OFFSET(f[1], dword, 6));
    e.mov(e.edx, PPCFIELD_PTR(r[4]));
    // can probably make this just 0x800f
    e.and_(e.eax, 0xFFFF800F);
    e.add(e.edx, 0x3fe);
    e.shl(e.edx, 4);

    e.or_(e.eax, e.edx);
    e.mov(PPCFIELD_PTR_SIZED_WITH_OFFSET(f[1], word, 6), e.ax);

  } else if (gf->ppc_builtin() ==
             PPCBuiltin::red_dead_redemption_freqcall_824EB0C8) {
    e.mov(e.rdx, e.GetContextReg());
    e.mov(e.r8, e.GetMembaseReg());
    e.CallNativeSafe(red_dead_redemption_freqcall_824EB0C8_thunk);
  } else if (gf->ppc_builtin() ==
             PPCBuiltin::red_dead_redemption_freqcall_824EAF38) {
    e.mov(e.rdx, e.GetContextReg());
    e.mov(e.r8, e.GetMembaseReg());
    e.CallNativeSafe(red_dead_redemption_freqcall_824EAF38_thunk);
  } else if (gf->ppc_builtin() ==
             PPCBuiltin::red_dead_redemption_switch_func_823C59F8) {
    Xbyak::Label skip;
    be.load_gpreg_arg(3, INT32_TYPE, be->ecx);
    be.load_gpreg_arg(4, INT32_TYPE, be->edx);
    be.load_gpreg_arg(5, INT32_TYPE, be->r8d);
    translate_address_in_register(e, be->rcx);

    call_named_thunk(be, "match_red_dead_redemption_switch_func_823C59F8_impl",
                     match_red_dead_redemption_switch_func_823C59F8_impl);

  } else if (gf->ppc_builtin() == PPCBuiltin::Ceilf1) {
    e.vroundsd(e.xmm0,
               e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, f[1])],
               10);
    e.vmovsd(e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, f[1])],
             e.xmm0);
  } else if (gf->ppc_builtin() == PPCBuiltin::uint64_to_double_variant1) {
    /*
            mov       rdx, 0xfffffffffffff                          #12.64
        lzcnt     rax, rdi                                      #9.12
        mov       ecx, eax                                      #10.32
        neg       rax                                           #12.24
        shl       rdi, cl                                       #10.32
        add       rax, 1086                                     #12.24
        mov       rsi, rdi                                      #12.58
        shr       rsi, 11                                       #12.58
        shl       rax, 52                                       #12.42
        and       rsi, rdx                                      #12.64
        or        rax, rsi                                      #12.64
        test      rdi, rdi                                      #12.7
        cmove     rax, rdi                                      #12.7
        ret                                                     #14.11
    */
    e.mov(e.rdx, PPCFIELD_PTR(r[3]));
    e.mov(e.r9, 0xfffffffffffff);

    e.lzcnt(e.rcx, e.rdx);
    e.mov(e.ebp, e.ecx);
    e.neg(e.rcx);
    e.shl(e.rdx, e.cl);
    e.add(e.rcx, 1086);
    e.mov(e.r8, e.rdx);
    e.shr(e.r8, 11);
    e.shl(e.rcx, 52);
    e.and_(e.r8, e.r9);
    e.or_(e.rcx, e.r8);
    e.test(e.rdx, e.rdx);
    e.cmove(e.rcx, e.rdx);
    e.mov(PPCFIELD_PTR(f[1]), e.rcx);

  } else if (gf->ppc_builtin() == PPCBuiltin::return_zero) {
    e.mov(PPCFIELD_PTR_SIZED(r[3], qword), 0);

  }

  else if (false&&gf->ppc_builtin() == PPCBuiltin::bungie_3d_normalize1) {
#if 0
    e.mov(e.r8d, e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);

    translate_address_in_register(e, e.r8);
    e.mov(e.rdx, e.GetContextReg());
    e.CallNativeSafe(&EmulateBungieNormalize3d);
#else
    e.mov(e.ecx, e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);
    translate_address_in_register(e, e.rcx);
    call_named_thunk(be, "bungie_normalize_thunk", EmulateBungieNormalize3d);
    e.vmovsd(PPCFIELD_PTR(f[1]), e.xmm0);
#endif

  } else if (false && gf->ppc_builtin() == PPCBuiltin::Memcpy_standard_variant1) {
    // rdx = arg0
    // r8  = arg1
    // r9  = arg2
    const Value* nbytes =
        hunt_context_store_backwards(i, offsetof(ppc::PPCContext_s, r[5]));

    if (!nbytes || !nbytes->IsConstant() ||
        !emit_memcpy_const_size(e, i, nbytes->constant.u32)) {
#if 0
      e.mov(e.edx,
            e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);
      e.mov(e.r8d,
            e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[4])]);
      e.mov(e.r9d,
            e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[5])]);
      translate_address_in_register(e, e.rdx);
      translate_address_in_register(e, e.r8);

      e.CallNativeSafe(&memcpy_forwarder);
#else
      e.mov(e.ecx,
            e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);
      e.mov(e.edx,
            e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[4])]);
      e.mov(e.r8d,
            e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[5])]);
      translate_address_in_register(e, e.rcx);
      translate_address_in_register(e, e.rdx);
      call_named_thunk(be, "ppc_thunk_memcpy",
                       GetProcAddress(GetModuleHandleA("ntdll.dll"), "memcpy"));

      //  e.CallNativeSafe(&memcpy_forwarder);
#endif
    }

  } else if (gf->ppc_builtin() == PPCBuiltin::bungie_data_iter_func1 ||
             gf->ppc_builtin() == PPCBuiltin::bungie_data_iter_func1_reach) {
    /*
      referencing the eldewrito halo 3 database for this
    */

    /*
    signed int __cdecl data_next_absolute_index(data_array *array, signed int
index)
{
signed int result; // eax
__int32 first_unallocated; // esi

result = index;
if ( index < 0 )
  return -1;
first_unallocated = array->first_unallocated;
if ( index >= first_unallocated )
  return -1;
while ( !((1 << (result & 0x1F)) & array->active_indices[result >> 5]) )
{
  if ( ++result >= first_unallocated )
    return -1;
}
return result;
}

 iVar2 = iVar1;
 if ((-1 < iVar3) && (iVar3 < *(int32_t*)(param_1 + 0x38))) {
    iVar4 = (int64_t)(int32_t)*(uint32_t*)(param_1 + 0x24) * (int64_t)iVar3;
    iVar2 = param_2;
    while (dataCacheBlockTouch(iVar4 + (uint64_t)*(uint32_t*)(param_1 + 0x44)),
(1 << (uint64_t)((uint32_t)iVar2 & 0x1f) &
*(uint32_t*)(((int32_t)(uint32_t)iVar2 >> 5) * 4 + *(int32_t*)(param_1 + 0x48)))
== 0) { iVar2 = iVar2 + 1; iVar4 = iVar4 + (uint64_t)*(uint32_t*)(param_1 +
0x24); if (*(int32_t*)(param_1 + 0x38) <= (int32_t)iVar2) { return iVar1;
       }
    }
 }
    */
    // array in ecx
    // auto offs = be.new_named_function(
    //   "bungie_data_iter_func1", [&be, &gf](Xbyak::CodeGenerator& e) {
    Xbyak::Label end, end_neg1_result, looper;

    e.mov(e.rcx,
          e.ptr[be->GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);
    e.mov(e.rdx,
          e.ptr[be->GetContextReg() + offsetof(ppc::PPCContext_s, r[4])]);

    translate_address_in_register(e, e.rcx);
    e.movbe(e.ebp, e.ptr[e.rcx + 0x38]);
    /*
    lwz       r10, 0x24(r11)
li        r7, 1
lwz       r9, 0x48(r11)
lwz       r8, 0x44(r11)
mullw     r11, r10, r4
    */
    e.movbe(e.r9d, e.ptr[e.rcx + 0x48]);

    e.test(e.edx, e.edx);
    e.js(end_neg1_result);
    e.cmp(e.edx, e.ebp);
    e.jge(end_neg1_result);

    translate_address_in_register(e, e.r9);
    /*
     (1 << (uint64_t)((uint32_t)iVar2 & 0x1f) &
     *(uint32_t*)(((int32_t)(uint32_t)iVar2 >> 5) * 4 +
     *(int32_t*)(param_1 + 0x48))) == 0) { iVar2 = iVar2 + 1; iVar4 =
     iVar4 + (uint64_t)*(uint32_t*)(param_1 + 0x24); if
     (*(int32_t*)(param_1 + 0x38) <= (int32_t)iVar2) { return iVar1;
       }
    */
    e.L(looper);
    e.mov(e.eax, e.edx);
    e.shr(e.eax, 5);
    e.movbe(e.eax, e.ptr[e.rax * 4 + e.r9]);
    e.bt(e.eax, e.edx);
    e.jc(end);
    e.add(e.edx, 1);

    e.cmp(e.edx, e.ebp);
    e.jg(end);
    e.jmp(looper);

    e.L(end_neg1_result);
    e.mov(e.edx, -1);
    e.jmp(end);

    e.L(end);
    /*
        is this necessary?
    */
    e.movsxd(e.rdx, e.edx);
    e.mov(e.ptr[be->GetContextReg() + offsetof(ppc::PPCContext_s, r[3])],
          e.rdx);
    //  e.ret();
    //  });
    // be->call((void*)offs);
  } else if ((gf->ppc_builtin() ==
                  PPCBuiltin::red_dead_redemption_tinyfunc_823DA208 ||
              gf->ppc_builtin() ==
                  PPCBuiltin::red_dead_redemption_tinyfunc_823da230) &&
             !(i->flags & hir::CALL_POSSIBLE_RETURN)) {
    /*
       int32_t *in_r13;

   (**(code**)(**(int32_t**)(*in_r13 + 0x78) + 8))(*(int32_t**)(*in_r13 + 0x78),
   param_1, 0x10, 0); return;
    */
    e.mov(e.edx, PPCFIELD_PTR(r[13]));

    e.mov(e.rcx, PPCFIELD_PTR(r[3]));
    e.mov(e.eax, 0x10);
    e.mov(PPCFIELD_PTR(r[4]), e.rcx);
    translate_address_in_register(e, e.rdx);
    e.xor_(e.ecx, e.ecx);

    // r13ptr in rdx now
    e.mov(PPCFIELD_PTR(r[5]), e.rax);
    e.movbe(e.edx, e.ptr[e.rdx]);
    e.mov(PPCFIELD_PTR(r[6]), e.rcx);
    translate_address_in_register(e, e.rdx);

    e.movbe(e.eax, e.ptr[e.rdx + 0x78]);
    /*
             lwz       r11, 0(r13)
.text:823DA20C 39 40 00 78                 li        r10, 0x78 # 'x'
.text:823DA210 7C 64 1B 78                 mr        r4, r3
.text:823DA214 38 C0 00 00                 li        r6, 0
.text:823DA218 38 A0 00 10                 li        r5, 0x10
.text:823DA21C 7C 6A 58 2E                 lwzx      r3, r10, r11
.text:823DA220 81 23 00 00                 lwz       r9, 0(r3)
.text:823DA224 81 09 00 08                 lwz       r8, 8(r9)
.text:823DA228 7D 09 03 A6                 mtspr     CTR, r8
.text:823DA22C 4E 80 04 20                 bctr
    */
    e.mov(e.edx, e.eax);
    translate_address_in_register(e, e.rax);
    e.mov(PPCFIELD_PTR(r[3]), e.rdx);
    // rax now contains r13+0x78 ptr
    e.movbe(e.ecx, e.ptr[e.rax]);  // load vftbl
    translate_address_in_register(e, e.rcx);
    e.movbe(e.eax, e.ptr[e.rcx + 8]);

    e.CallIndirect(i, e.rax);

  } else if (gf->ppc_builtin() ==
             PPCBuiltin::red_dead_redemption_tinyfunc_823DA2F8) {
    Xbyak::Label end;
    e.mov(e.eax, PPCFIELD_PTR(r[3]));
    // (**(code**)(**(int32_t**)(*in_r13 + 0x78) + 0x10))(*(int32_t**)(*in_r13 +
    // 0x78), param_1);
    e.mov(e.ecx, PPCFIELD_PTR(r[13]));
    translate_address_in_register(e, e.rcx);
    e.test(e.eax, e.eax);
    e.jz(end);
    //*r13
    e.movbe(e.edx, e.ptr[e.rcx]);
    e.mov(PPCFIELD_PTR(r[4]), e.eax);
    translate_address_in_register(e, e.rdx);
    e.movbe(e.ecx, e.ptr[e.rdx + 0x78]);
    e.mov(PPCFIELD_PTR(r[3]), e.ecx);
    translate_address_in_register(e, e.rcx);
    e.movbe(e.eax, e.ptr[e.rcx]);
    translate_address_in_register(e, e.rax);
    e.movbe(e.eax, e.ptr[e.rax + 0x10]);
    e.CallIndirect(i, e.rax);
    e.L(end);
    /*
    lwz       r11, 0(r13)
li        r10, 0x78 # 'x'
lwzx      r3, r10, r11
lwz       r9, 0(r3)
lwz       r8, 0x10(r9)
mtspr     CTR, r8
bctr
    */

  } else if (gfid == PPCBuiltin::popcount_uint32) {
    e.popcnt(e.eax, PPCFIELD_PTR(r[3]));
    e.mov(PPCFIELD_PTR(r[3]), e.eax);
  } else if (false && gf->ppc_builtin() == PPCBuiltin::Memmove_standard) {
    // rdx = arg0
    // r8  = arg1
    // r9  = arg2
    e.mov(e.edx, e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);
    e.mov(e.r8d, e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[4])]);
    e.mov(e.r9d, e.ptr[e.GetContextReg() + offsetof(ppc::PPCContext_s, r[5])]);
    translate_address_in_register(e, e.rdx);
    translate_address_in_register(e, e.r8);

    e.CallNativeSafe(&memmove_forwarder);

  } else if (gf->ppc_builtin() == PPCBuiltin::strncmp_variant1) {
    emit_strncmp(be);
  } else if (gf->ppc_builtin() == PPCBuiltin::bungie_data_iter_increment_h3 ||
             gf->ppc_builtin() ==
                 PPCBuiltin::bungie_data_iter_increment_reach) {
    // auto offs = be.new_named_function("bungie_data_iter_incr", [&be,
    // &gf](Xbyak::CodeGenerator& e) { e.push(e.r8);
    e.mov(e.r8, e.ptr[be->GetContextReg() + offsetof(ppc::PPCContext_s, r[3])]);
    translate_address_in_register(e, e.r8);
    bool isreach =
        gf->ppc_builtin() == PPCBuiltin::bungie_data_iter_increment_reach;
    Xbyak::Label end, end_neg1_result, looper, setup_end_of_iter, really_done;

    e.movbe(e.ecx, e.ptr[e.r8]);

    e.movbe(e.edx, e.ptr[e.r8 + 8]);

    translate_address_in_register(e, e.rcx);
    e.add(e.edx, 1);
    e.movbe(e.ebp, e.ptr[e.rcx + 0x38]);

    e.movbe(e.r9d, e.ptr[e.rcx + 0x48]);

    e.test(e.edx, e.edx);
    e.js(end_neg1_result);
    e.cmp(e.edx, e.ebp);
    e.jge(end_neg1_result);

    translate_address_in_register(e, e.r9);
    e.L(looper);
    e.mov(e.eax, e.edx);
    e.shr(e.eax, 5);
    e.movbe(e.eax, e.ptr[e.rax * 4 + e.r9]);
    e.bt(e.eax, e.edx);
    e.jc(end);
    e.add(e.edx, 1);

    e.cmp(e.edx, e.ebp);
    e.jg(end);
    e.jmp(looper);

    e.L(end);

    e.cmp(e.edx, -1);
    e.je(setup_end_of_iter);

    e.movbe(e.r9d, e.ptr[e.rcx + (isreach ? 0x20 : 0x24)]);
    e.movbe(e.ebp, e.ptr[e.rcx + 0x44]);
    e.movbe(e.ptr[e.r8 + 8], e.edx);
    e.push(e.rdx);
    e.imul(e.edx, e.r9d);
    e.add(e.edx, e.ebp);
    // edx is our result!!
    e.mov(e.ebp, e.edx);
    translate_address_in_register(e, e.rbp);
    e.movbe(e.ax, e.ptr[e.rbp]);
    e.shl(e.eax, 16);
    e.pop(e.rbp);
    e.or_(e.eax, e.ebp);
    e.movbe(e.ptr[e.r8 + 4], e.eax);
    e.jmp(really_done);
    e.L(end_neg1_result);
    e.L(setup_end_of_iter);

    e.xor_(e.edx, e.edx);

    e.mov(e.eax, e.ptr[e.rcx + (isreach ? 0x28 : 0x20)]);
    e.mov(e.ebp, -1);
    e.mov(e.ptr[e.r8 + 4], e.ebp);
    e.mov(e.ptr[e.r8 + 8], e.eax);

    e.L(really_done);
    e.mov(e.ptr[be->GetContextReg() + offsetof(ppc::PPCContext_s, r[3])],
          e.rdx);
    // e.ret();
    //   });
    // be->call((void*)offs);
  } else
    e.Call(i, gf);
}

// ============================================================================
// OPCODE_CALL
// ============================================================================
struct CALL : Sequence<CALL, I<OPCODE_CALL, VoidOp, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    assert_true(i.src1.value->is_guest());
    do_guest_call_or_builtin(e, i.instr,
                             static_cast<GuestFunction*>(i.src1.value));
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CALL, CALL);

// ============================================================================
// OPCODE_CALL_TRUE
// ============================================================================
struct CALL_TRUE_I8
    : Sequence<CALL_TRUE_I8, I<OPCODE_CALL_TRUE, VoidOp, I8Op, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.value->is_guest());
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    do_guest_call_or_builtin(e, i.instr,
                             static_cast<GuestFunction*>(i.src2.value));
    e.L(skip);
  }
};
struct CALL_TRUE_I16
    : Sequence<CALL_TRUE_I16, I<OPCODE_CALL_TRUE, VoidOp, I16Op, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.value->is_guest());
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    do_guest_call_or_builtin(e, i.instr,
                             static_cast<GuestFunction*>(i.src2.value));
    e.L(skip);
  }
};
struct CALL_TRUE_I32
    : Sequence<CALL_TRUE_I32, I<OPCODE_CALL_TRUE, VoidOp, I32Op, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.value->is_guest());
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    do_guest_call_or_builtin(e, i.instr,
                             static_cast<GuestFunction*>(i.src2.value));
    e.L(skip);
  }
};
struct CALL_TRUE_I64
    : Sequence<CALL_TRUE_I64, I<OPCODE_CALL_TRUE, VoidOp, I64Op, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.value->is_guest());
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    do_guest_call_or_builtin(e, i.instr,
                             static_cast<GuestFunction*>(i.src2.value));
    e.L(skip);
  }
};
struct CALL_TRUE_F32
    : Sequence<CALL_TRUE_F32, I<OPCODE_CALL_TRUE, VoidOp, F32Op, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.value->is_guest());
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    do_guest_call_or_builtin(e, i.instr,
                             static_cast<GuestFunction*>(i.src2.value));
    e.L(skip);
  }
};
struct CALL_TRUE_F64
    : Sequence<CALL_TRUE_F64, I<OPCODE_CALL_TRUE, VoidOp, F64Op, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    assert_true(i.src2.value->is_guest());
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip);
    do_guest_call_or_builtin(e, i.instr,
                             static_cast<GuestFunction*>(i.src2.value));
    e.L(skip);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CALL_TRUE, CALL_TRUE_I8, CALL_TRUE_I16,
                     CALL_TRUE_I32, CALL_TRUE_I64, CALL_TRUE_F32,
                     CALL_TRUE_F64);

// ============================================================================
// OPCODE_CALL_INDIRECT
// ============================================================================
struct CALL_INDIRECT
    : Sequence<CALL_INDIRECT, I<OPCODE_CALL_INDIRECT, VoidOp, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.CallIndirect(i.instr, i.src1);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CALL_INDIRECT, CALL_INDIRECT);

// ============================================================================
// OPCODE_CALL_INDIRECT_TRUE
// ============================================================================
struct CALL_INDIRECT_TRUE_I8
    : Sequence<CALL_INDIRECT_TRUE_I8,
               I<OPCODE_CALL_INDIRECT_TRUE, VoidOp, I8Op, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip, CodeGenerator::T_NEAR);
    e.CallIndirect(i.instr, i.src2);
    e.L(skip);
  }
};
struct CALL_INDIRECT_TRUE_I16
    : Sequence<CALL_INDIRECT_TRUE_I16,
               I<OPCODE_CALL_INDIRECT_TRUE, VoidOp, I16Op, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip, CodeGenerator::T_NEAR);
    e.CallIndirect(i.instr, i.src2);
    e.L(skip);
  }
};
struct CALL_INDIRECT_TRUE_I32
    : Sequence<CALL_INDIRECT_TRUE_I32,
               I<OPCODE_CALL_INDIRECT_TRUE, VoidOp, I32Op, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip, CodeGenerator::T_NEAR);
    e.CallIndirect(i.instr, i.src2);
    e.L(skip);
  }
};
struct CALL_INDIRECT_TRUE_I64
    : Sequence<CALL_INDIRECT_TRUE_I64,
               I<OPCODE_CALL_INDIRECT_TRUE, VoidOp, I64Op, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip, CodeGenerator::T_NEAR);
    e.CallIndirect(i.instr, i.src2);
    e.L(skip);
  }
};
struct CALL_INDIRECT_TRUE_F32
    : Sequence<CALL_INDIRECT_TRUE_F32,
               I<OPCODE_CALL_INDIRECT_TRUE, VoidOp, F32Op, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip, CodeGenerator::T_NEAR);
    e.CallIndirect(i.instr, i.src2);
    e.L(skip);
  }
};
struct CALL_INDIRECT_TRUE_F64
    : Sequence<CALL_INDIRECT_TRUE_F64,
               I<OPCODE_CALL_INDIRECT_TRUE, VoidOp, F64Op, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    Xbyak::Label skip;
    e.jz(skip, CodeGenerator::T_NEAR);
    e.CallIndirect(i.instr, i.src2);
    e.L(skip);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CALL_INDIRECT_TRUE, CALL_INDIRECT_TRUE_I8,
                     CALL_INDIRECT_TRUE_I16, CALL_INDIRECT_TRUE_I32,
                     CALL_INDIRECT_TRUE_I64, CALL_INDIRECT_TRUE_F32,
                     CALL_INDIRECT_TRUE_F64);

// ============================================================================
// OPCODE_CALL_EXTERN
// ============================================================================
struct CALL_EXTERN
    : Sequence<CALL_EXTERN, I<OPCODE_CALL_EXTERN, VoidOp, SymbolOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.CallExtern(i.instr, i.src1.value);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_CALL_EXTERN, CALL_EXTERN);

// ============================================================================
// OPCODE_RETURN
// ============================================================================
struct RETURN : Sequence<RETURN, I<OPCODE_RETURN, VoidOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    // If this is the last instruction in the last block, just let us
    // fall through.
    if (i.instr->next || i.instr->block->next) {
      e.jmp(e.epilog_label(), CodeGenerator::T_NEAR);
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_RETURN, RETURN);

// ============================================================================
// OPCODE_RETURN_TRUE
// ============================================================================
struct RETURN_TRUE_I8
    : Sequence<RETURN_TRUE_I8, I<OPCODE_RETURN_TRUE, VoidOp, I8Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jnz(e.epilog_label(), CodeGenerator::T_NEAR);
  }
};
struct RETURN_TRUE_I16
    : Sequence<RETURN_TRUE_I16, I<OPCODE_RETURN_TRUE, VoidOp, I16Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jnz(e.epilog_label(), CodeGenerator::T_NEAR);
  }
};
struct RETURN_TRUE_I32
    : Sequence<RETURN_TRUE_I32, I<OPCODE_RETURN_TRUE, VoidOp, I32Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jnz(e.epilog_label(), CodeGenerator::T_NEAR);
  }
};
struct RETURN_TRUE_I64
    : Sequence<RETURN_TRUE_I64, I<OPCODE_RETURN_TRUE, VoidOp, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jnz(e.epilog_label(), CodeGenerator::T_NEAR);
  }
};
struct RETURN_TRUE_F32
    : Sequence<RETURN_TRUE_F32, I<OPCODE_RETURN_TRUE, VoidOp, F32Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    e.jnz(e.epilog_label(), CodeGenerator::T_NEAR);
  }
};
struct RETURN_TRUE_F64
    : Sequence<RETURN_TRUE_F64, I<OPCODE_RETURN_TRUE, VoidOp, F64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    e.jnz(e.epilog_label(), CodeGenerator::T_NEAR);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_RETURN_TRUE, RETURN_TRUE_I8, RETURN_TRUE_I16,
                     RETURN_TRUE_I32, RETURN_TRUE_I64, RETURN_TRUE_F32,
                     RETURN_TRUE_F64);

// ============================================================================
// OPCODE_SET_RETURN_ADDRESS
// ============================================================================
struct SET_RETURN_ADDRESS
    : Sequence<SET_RETURN_ADDRESS,
               I<OPCODE_SET_RETURN_ADDRESS, VoidOp, I64Op>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.SetReturnAddress(i.src1.constant());
  }
};
EMITTER_OPCODE_TABLE(OPCODE_SET_RETURN_ADDRESS, SET_RETURN_ADDRESS);

// ============================================================================
// OPCODE_BRANCH
// ============================================================================
struct BRANCH : Sequence<BRANCH, I<OPCODE_BRANCH, VoidOp, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.jmp(i.src1.value->name, e.T_NEAR);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_BRANCH, BRANCH);

// ============================================================================
// OPCODE_BRANCH_TRUE
// ============================================================================
struct BRANCH_TRUE_I8
    : Sequence<BRANCH_TRUE_I8, I<OPCODE_BRANCH_TRUE, VoidOp, I8Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    EmitFusedBranch(e, i);
  }
};
struct BRANCH_TRUE_I16
    : Sequence<BRANCH_TRUE_I16, I<OPCODE_BRANCH_TRUE, VoidOp, I16Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    EmitFusedBranch(e, i);
  }
};
struct BRANCH_TRUE_I32
    : Sequence<BRANCH_TRUE_I32, I<OPCODE_BRANCH_TRUE, VoidOp, I32Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    EmitFusedBranch(e, i);
  }
};
struct BRANCH_TRUE_I64
    : Sequence<BRANCH_TRUE_I64, I<OPCODE_BRANCH_TRUE, VoidOp, I64Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    EmitFusedBranch(e, i);
  }
};
struct BRANCH_TRUE_F32
    : Sequence<BRANCH_TRUE_F32, I<OPCODE_BRANCH_TRUE, VoidOp, F32Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    if (i.instr->prev && i.instr->prev->opcode == &OPCODE_IS_TRUE_info &&
        i.instr->prev->dest == i.src1.value) {
      e.jnz(i.src2.value->name, e.T_NEAR);
    } else if (i.instr->prev &&
               i.instr->prev->opcode == &OPCODE_IS_FALSE_info &&
               i.instr->prev->dest == i.src1.value) {
      e.jz(i.src2.value->name, e.T_NEAR);
    } else {
      e.vptest(i.src1, i.src1);
      e.jnz(i.src2.value->name, e.T_NEAR);
    }
  }
};
struct BRANCH_TRUE_F64
    : Sequence<BRANCH_TRUE_F64, I<OPCODE_BRANCH_TRUE, VoidOp, F64Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    if (i.instr->prev && i.instr->prev->opcode == &OPCODE_IS_TRUE_info &&
        i.instr->prev->dest == i.src1.value) {
      e.jnz(i.src2.value->name, e.T_NEAR);
    } else if (i.instr->prev &&
               i.instr->prev->opcode == &OPCODE_IS_FALSE_info &&
               i.instr->prev->dest == i.src1.value) {
      e.jz(i.src2.value->name, e.T_NEAR);
    } else {
      e.vptest(i.src1, i.src1);
      e.jnz(i.src2.value->name, e.T_NEAR);
    }
  }
};
EMITTER_OPCODE_TABLE(OPCODE_BRANCH_TRUE, BRANCH_TRUE_I8, BRANCH_TRUE_I16,
                     BRANCH_TRUE_I32, BRANCH_TRUE_I64, BRANCH_TRUE_F32,
                     BRANCH_TRUE_F64);

// ============================================================================
// OPCODE_BRANCH_FALSE
// ============================================================================
struct BRANCH_FALSE_I8
    : Sequence<BRANCH_FALSE_I8, I<OPCODE_BRANCH_FALSE, VoidOp, I8Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jz(i.src2.value->name, e.T_NEAR);
  }
};
struct BRANCH_FALSE_I16
    : Sequence<BRANCH_FALSE_I16,
               I<OPCODE_BRANCH_FALSE, VoidOp, I16Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jz(i.src2.value->name, e.T_NEAR);
  }
};
struct BRANCH_FALSE_I32
    : Sequence<BRANCH_FALSE_I32,
               I<OPCODE_BRANCH_FALSE, VoidOp, I32Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jz(i.src2.value->name, e.T_NEAR);
  }
};
struct BRANCH_FALSE_I64
    : Sequence<BRANCH_FALSE_I64,
               I<OPCODE_BRANCH_FALSE, VoidOp, I64Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.test(i.src1, i.src1);
    e.jz(i.src2.value->name, e.T_NEAR);
  }
};
struct BRANCH_FALSE_F32
    : Sequence<BRANCH_FALSE_F32,
               I<OPCODE_BRANCH_FALSE, VoidOp, F32Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    e.jz(i.src2.value->name, e.T_NEAR);
  }
};
struct BRANCH_FALSE_F64
    : Sequence<BRANCH_FALSE_F64,
               I<OPCODE_BRANCH_FALSE, VoidOp, F64Op, LabelOp>> {
  static void Emit(X64Emitter& e, const EmitArgType& i) {
    e.vptest(i.src1, i.src1);
    e.jz(i.src2.value->name, e.T_NEAR);
  }
};
EMITTER_OPCODE_TABLE(OPCODE_BRANCH_FALSE, BRANCH_FALSE_I8, BRANCH_FALSE_I16,
                     BRANCH_FALSE_I32, BRANCH_FALSE_I64, BRANCH_FALSE_F32,
                     BRANCH_FALSE_F64);

}  // namespace x64
}  // namespace backend
}  // namespace cpu
}  // namespace xe