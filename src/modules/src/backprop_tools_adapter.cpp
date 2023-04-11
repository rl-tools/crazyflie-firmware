
#include "backprop_tools_adapter.h"

#include <layer_in_c/operations/arm.h>
#include <layer_in_c/nn/layers/dense/operations_arm/opt.h>
#include <layer_in_c/nn/layers/dense/operations_arm/dsp.h>
#include <layer_in_c/nn_models/mlp/operations_generic.h>
#include <test_layer_in_c_nn_models_mlp_persist_code.h> // attention: the array literals have to be declared "const"
#include <test_layer_in_c_nn_models_mlp_evaluation.h> // attention: the array literals have to be declared "const"


namespace lic = layer_in_c;

using DEV_SPEC = lic::devices::DefaultARMSpecification;
using DEVICE = lic::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using TI = typename mlp_1::SPEC::TI;
using DTYPE = typename mlp_1::SPEC::T;

DTYPE input_layer_output_memory[mlp_1::SPEC::OUTPUT_DIM];
lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> input_layer_output = {(DTYPE*)input_layer_output_memory};

void backprop_tools_run(float* output_mem){
    DTYPE buffer_tick_memory[mlp_1::SPEC::HIDDEN_DIM];
    DTYPE buffer_tock_memory[mlp_1::SPEC::HIDDEN_DIM];
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tick = {(DTYPE*)buffer_tick_memory};
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tock = {(DTYPE*)buffer_tock_memory};

    decltype(mlp_1::mlp)::template Buffers<1> buffers = {buffer_tick, buffer_tock};
    lic::MatrixDynamic<lic::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::OUTPUT_DIM, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)output_mem};
    auto input_sample = lic::row(device, input::container, 0);
    int iterations = 10;
    for(int iteration_i = 0; iteration_i < iterations; iteration_i++){
        lic::evaluate(device, mlp_1::mlp, input_sample, output, buffers);
    }
}