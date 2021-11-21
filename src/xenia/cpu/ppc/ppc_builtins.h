#ifndef XENIA_CPU_PPC_PPC_BUILTIN_H_
#define XENIA_CPU_PPC_PPC_BUILTIN_H_

#include <cstdint>
#include <cstdlib>
namespace xe {
namespace cpu {
namespace ppc {

	enum class PPCBuiltin : uint32_t {
		Unclassified,
		None,
		Memset,
		Floorf1, //floor, variant 1
		Ceilf1, //ceil, variant 1
		chkstk1, //rtl checkstack, variant 1
		bungie_3d_normalize1, //bungie 3d normalize, variant 1
		Memcpy_standard_variant1, //c standard library memcpy prototype
		strncmp_variant1,
		Memmove_standard,
		bungie_data_iter_func1,
		bungie_data_iter_increment_h3,
		bungie_data_iter_func1_reach,
		bungie_data_iter_increment_reach,

	
		strcmpi_variant1,
		uint64_to_double_variant1,
		stricmp_variant1,
		strchr_variant1,
		strrchr_variant1,
		wcslen_variant1,
		get_exp_variant1,
		set_exp_variant1,

		blrfunc, //function that just does blr and nothing else, remove it in dead code elimination hir pass
		/*
			red dead redemption specific ones start here
		*/

		red_dead_redemption_tinyfunc_823DA208, // called 7056 times
		red_dead_redemption_tinyfunc_823DA2F8, // called 4096 times
		red_dead_redemption_tinyfunc_823DA328, // called 3074 times
		red_dead_redemption_tinyfunc_823da230, //called 1162 times
		red_dead_redemption_tinyfunc_823da290, //called 665 times
		red_dead_redemption_tinyfunc_823da258, //called 427 times

		red_dead_redemption_freqcall_824EAF38,
		red_dead_redemption_freqcall_824EB0C8,

		//these are reserved from our call counting, the most frequently called functions
		red_dead_redemption_tinyfunc_calledbillions, //may be directx lib related?

		red_dead_redemption_sub_82EB1110,
		red_dead_redemption_sub_82A565E8,
		//in the top 20 functions and pretty easy to implement
		red_dead_redemption_switch_func_823C59F8,

		rdr_hashcode_function, //may appear in other RAGE engine games. 
		//takes two params, first one char*, second one a seed. seed is almost always zero.
		//called an insane number of times per frame, likely used for some sort of dictionary lookup

		flush_cached_mem_d3dlib,
		freqcall_c_allocator_, //rdr 82BC6188. actually not called very frequently. woops. this does appear in other xbox games though
		d3d_device_set_pix_shader_constant_fn, //one of the most frequently called functions in rdr, likely called in other games
		d3d_device_set_vert_shader_constant_fn, //one of the most frequently called functions in rdr, likely called in other games
		/*
			savegplr
		*/
		savegplr,
		savegplr15,
		savegplr16,
		savegplr17,
		savegplr18,
		savegplr19,
		savegplr20,
		savegplr21,
		savegplr22,
		savegplr23,
		savegplr24,
		savegplr25,
		savegplr26,
		savegplr27,
		savegplr28,
		savegplr29,
		savegplr30,
		savegplr31,

		/*
			restgplr
		*/

		restgplr,
		restgplr15,
		restgplr16,
		restgplr17,
		restgplr18,
		restgplr19,
		restgplr20,
		restgplr21,
		restgplr22,
		restgplr23,
		restgplr24,
		restgplr25,
		restgplr26,
		restgplr27,
		restgplr28,
		restgplr29,
		restgplr30,
		restgplr31,

		//prototype : unsigned popcnt(unsigned w)
		popcount_uint32,
			return_zero, //li r3, 0 blr
		d3d_blocker_check, //h3 825A7EC8
			//most called func in h3 by a mile
	};

	PPCBuiltin classify_function_at(uint8_t* fn, uint8_t* module_base, Module* rm);

	}
}  // namespace cpu
}  // namespace xe

#endif
