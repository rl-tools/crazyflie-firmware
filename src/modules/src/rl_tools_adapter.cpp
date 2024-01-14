
#include "rl_tools_adapter.h"

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/dense/operations_arm/dsp.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <nn_models_sequential_persist_code.h>


namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using TI = typename rl_tools_export::model::MODEL::TI;
using T = typename rl_tools_export::model::MODEL::T;
rl_tools_export::model::MODEL::template Buffer<1, rlt::MatrixStaticTag> buffers;
rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, rl_tools_export::model::MODEL::OUTPUT_DIM>> output;

void rl_tools_init(){
    rlt::malloc(device, buffers);
    rlt::malloc(device, output);
}
void rl_tools_deinit(){
    rlt::free(device, buffers);
    rlt::free(device, output);
}

float rl_tools_run(float* output_mem){
    auto input_sample = rlt::row(device, rl_tools_export::input::container, 0);
    int iterations = 10;
    for(TI iteration_i = 0; iteration_i < iterations; iteration_i++){
        rlt::evaluate(device, rl_tools_export::model::model, input_sample, output, buffers);
    }
    auto output_wrapped = rlt::wrap<DEVICE, float, rl_tools_export::model::MODEL::OUTPUT_DIM>(device, output_mem);
    rlt::copy(device, device, output, output_wrapped);
    T acc = 0;
    for(TI output_i=0; output_i < rl_tools_export::model::MODEL::OUTPUT_DIM; output_i++){
        acc += rlt::math::abs(device.math, rlt::get(output, 0, output_i) - rlt::get(rl_tools_export::output::container, 0, output_i));
    }
    return acc;
}
