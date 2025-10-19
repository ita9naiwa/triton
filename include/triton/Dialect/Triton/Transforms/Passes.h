#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

// SM_120: Convert simple tt.load to descriptor-based loads to enable TMA
std::unique_ptr<mlir::Pass>
createConvertLoadToDescriptorSM120Pass(int computeCapability);

} // namespace triton
} // namespace mlir

#endif
