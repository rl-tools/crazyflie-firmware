#include "backprop_tools_adapter.h"

#include <layer_in_c/operations/dummy.h>
#include <layer_in_c/nn/layers/dense/operations_dummy.h>

namespace lic = layer_in_c;


using DTYPE = float;
using TI = unsigned;
using DEVICE = lic::devices::DefaultDummy;
constexpr TI INPUT_DIM = 64;
constexpr TI HIDDEN_DIM = 64;
constexpr TI N_LAYERS = 3;
constexpr TI OUTPUT_DIM = 10;


DEVICE device;

DTYPE input_mem[INPUT_DIM];
lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, INPUT_DIM>> input = {(DTYPE*)input_mem};
DTYPE layer_1_W_mem[HIDDEN_DIM][INPUT_DIM];
lic::Matrix<lic::matrix::Specification<DTYPE, TI, INPUT_DIM, HIDDEN_DIM>> layer_1_W = {(DTYPE*)layer_1_W_mem};
DTYPE layer_1_b_mem[HIDDEN_DIM];
lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, HIDDEN_DIM>> layer_1_b = {(DTYPE*)layer_1_b_mem};
DTYPE layer_1_output_mem[HIDDEN_DIM];
lic::Matrix<lic::matrix::Specification<DTYPE, TI, 1, HIDDEN_DIM>> layer_1_output = {(DTYPE*)layer_1_b_mem};

using LAYER_1_SPEC = lic::nn::layers::dense::Specification<DTYPE, TI, INPUT_DIM, HIDDEN_DIM, lic::nn::activation_functions::ActivationFunction::RELU>;
lic::nn::layers::dense::Layer<LAYER_1_SPEC> layer_1 = {layer_1_W, layer_1_b};

void backprop_tools_run(){
  lic::evaluate(device, layer_1, input, layer_1_output);
}