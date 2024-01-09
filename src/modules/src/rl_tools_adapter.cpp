
#include "rl_tools_adapter.h"

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/dense/operations_arm/dsp.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <nn_models_sequential_persist_code.h> // attention: the array literals have to be declared "const"
// #include <test_rl_tools_nn_models_mlp_evaluation.h> // attention: the array literals have to be declared "const"


namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using TI = typename rl_tools_export::model::MODEL::TI;
using T = typename rl_tools_export::model::MODEL::T;
rl_tools_export::model::MODEL::template DoubleBuffer<1, rlt::MatrixStaticTag> buffers;

// T input_layer_output_memory[rl_tools_export::model::MODEL::OUTPUT_DIM];
// rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::HIDDEN_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> input_layer_output = {(T*)input_layer_output_memory};
void rl_tools_init(){
    rlt::malloc(device, buffers);
}
void rl_tools_deinit(){
    rlt::free(device, buffers);
}

void rl_tools_run(float* output_mem){
    // T buffer_tick_memory[rl_tools_export::model::MODEL::HIDDEN_DIM];
    // T buffer_tock_memory[rl_tools_export::model::MODEL::HIDDEN_DIM];
    // rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::HIDDEN_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tick = {(T*)buffer_tick_memory};
    // rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::HIDDEN_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tock = {(T*)buffer_tock_memory};

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::OUTPUT_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)output_mem};
    auto input_sample = rlt::row(device, rl_tools_export::input::container, 0);
    int iterations = 10;
    for(int iteration_i = 0; iteration_i < iterations; iteration_i++){
        rlt::evaluate(device, rl_tools_export::model::model, input_sample, output, buffers);
    }
}
