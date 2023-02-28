
#include "backprop_tools_adapter.h"

#include <layer_in_c/operations/arm.h>
#include <layer_in_c/nn_models/mlp/operations_dummy.h>
#include <test_layer_in_c_nn_models_mlp_persist_code.h>
#include <test_layer_in_c_nn_models_mlp_evaluation.h>


namespace lic = layer_in_c;

using DEVICE = lic::devices::DefaultARM;
DEVICE device;
using TI = typename mlp_1::SPEC::TI;
using DTYPE = typename mlp_1::SPEC::T;

DTYPE input_layer_output_memory[mlp_1::SPEC::OUTPUT_DIM];
lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> input_layer_output = {(DTYPE*)input_layer_output_memory};

// DTYPE output_mem[mlp_1::SPEC::OUTPUT_DIM];
// lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::OUTPUT_DIM>> output = {(DTYPE*)output_mem};


void backprop_tools_run(float* output_mem){
    DTYPE buffer_tick_memory[mlp_1::SPEC::HIDDEN_DIM];
    DTYPE buffer_tock_memory[mlp_1::SPEC::HIDDEN_DIM];
    lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tick = {(DTYPE*)buffer_tick_memory};
    lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, mlp_1::SPEC::MEMORY_LAYOUT>> buffer_tock = {(DTYPE*)buffer_tock_memory};

    decltype(mlp_1::mlp)::template Buffers<1> buffers = {buffer_tick, buffer_tock};
    lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::OUTPUT_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)output_mem};
    // lic::evaluate(device, mlp_1::mlp.input_layer, input::matrix, output);
    // output_mem[0] = lic::get(mlp_1::mlp.input_layer.biases, 0, 0);
    lic::evaluate(device, mlp_1::mlp, input::matrix, output, buffers);
}