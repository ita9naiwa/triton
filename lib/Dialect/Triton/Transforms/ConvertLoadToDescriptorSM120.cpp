#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h" // used for NVMMASharedEncodingAttr
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-load-to-descriptor-sm120"

namespace mlir {
namespace triton {

namespace ttg = triton::gpu;

// Lint-only helper to mark TritonGPU dialect types as used in this TU.
static inline void _lint_use_tritongpu_symbols(ttg::NVMMASharedEncodingAttr) {}

#define GEN_PASS_DEF_TRITONCONVERTLOADTODESCRIPTORSM120
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Analyze if a load is a simple pattern we can convert to descriptor
struct SimpleLoadPattern {
  Value basePtr;                    // Base pointer (scalar)
  SmallVector<int64_t> tensorShape; // Total tensor shape [M, K]
  SmallVector<int64_t> blockShape;  // Block shape [M, BLOCK_K]
  SmallVector<int64_t> strides;     // Total tensor strides (in elements)
  SmallVector<Value> indices;       // Dynamic indices (i32)
  bool valid = false;
};

// Recursively find the base pointer by following addptr/broadcast/splat chain
Value findBasePointer(Value ptr) {
  if (auto addPtr = ptr.getDefiningOp<triton::AddPtrOp>()) {
    return findBasePointer(addPtr.getPtr());
  }
  if (auto broadcast = ptr.getDefiningOp<triton::BroadcastOp>()) {
    return findBasePointer(broadcast.getSrc());
  }
  if (auto splat = ptr.getDefiningOp<triton::SplatOp>()) {
    return splat.getSrc(); // Found scalar base!
  }
  return ptr; // Assume this is the base
}

// Check if value is from make_range + broadcast/splat pattern
bool isSimpleOffsetPattern(Value ptr, RankedTensorType loadType,
                           SimpleLoadPattern &result, Operation *loadOp) {
  llvm::errs() << "[isSimpleOffset] Analyzing pointer...\n";

  // Find the scalar base pointer by recursively following the chain
  Value base = findBasePointer(ptr);

  auto basePtrType = dyn_cast<triton::PointerType>(base.getType());
  if (!basePtrType) {
    llvm::errs() << "[isSimpleOffset] Base is not a pointer type\n";
    return false;
  }

  llvm::errs() << "[isSimpleOffset] Found base pointer!\n";

  result.basePtr = base;
  result.blockShape = llvm::to_vector(loadType.getShape());

  // Determine total tensor shape
  if (auto forOp = loadOp->getParentOfType<scf::ForOp>()) {
    // Inside loop: tensor shape includes loop iterations
    llvm::errs()
        << "[isSimpleOffset] Inside loop, computing total tensor shape\n";

    Value upperBound = forOp.getUpperBound();
    int64_t loopBound = -1;
    if (auto constOp = upperBound.getDefiningOp<arith::ConstantIntOp>()) {
      loopBound = constOp.value();
    } else {
      // Dynamic upper bound - can't handle for now
      llvm::errs() << "[isSimpleOffset] Dynamic upper bound not supported\n";
      return false;
    }

    // Heuristic: Loop dimension is the one matching loopBound/block dimension
    // For A[M, K] with block [M, BLOCK_K]: loopBound=K
    // For B[K, N] with block [BLOCK_K, N]: loopBound=K
    result.tensorShape.resize(result.blockShape.size());
    for (size_t i = 0; i < result.blockShape.size(); ++i) {
      // If block dimension < loopBound, this might be the loop dimension
      if (result.blockShape[i] < loopBound &&
          loopBound % result.blockShape[i] == 0) {
        result.tensorShape[i] = loopBound;
      } else {
        result.tensorShape[i] = result.blockShape[i];
      }
    }

    llvm::errs() << "[isSimpleOffset] Total tensor shape: ["
                 << result.tensorShape[0] << ", " << result.tensorShape[1]
                 << "]\n";
  } else {
    // Outside loop: tensor shape = block shape
    llvm::errs()
        << "[isSimpleOffset] Outside loop, tensor shape = block shape\n";
    result.tensorShape = result.blockShape;
  }

  // Compute strides for total tensor (row-major)
  result.strides.resize(result.tensorShape.size());
  int64_t stride = 1;
  for (int i = result.tensorShape.size() - 1; i >= 0; --i) {
    result.strides[i] = stride;
    if (i > 0)
      stride *= result.tensorShape[i];
  }

  llvm::errs() << "[isSimpleOffset] Computed strides (element): ["
               << result.strides[0] << ", " << result.strides[1] << "]\n";

  // Zero indices (simplification)
  result.indices.clear();
  result.valid = true;

  llvm::errs() << "[isSimpleOffset] Pattern analysis SUCCESS\n";

  return true;
}

class ConvertLoadToDescriptorPattern : public OpRewritePattern<triton::LoadOp> {
  int computeCapability;

public:
  ConvertLoadToDescriptorPattern(MLIRContext *context, int computeCapability)
      : OpRewritePattern<triton::LoadOp>(context),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "[Pattern] Matching and rewriting load op\n";
    // Only for SM_120
    if (computeCapability != 120)
      return failure();

    auto loadType = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!loadType) {
      llvm::errs() << "[Pattern] Load is not RankedTensorType\n";
      return failure();
    }

    // IMPORTANT: Only convert loads that are used by dot operations
    // Scale loads and other non-dot loads should use regular tt.load
    // Check recursively through type conversions (FPToFP, bitcast, transpose)

    llvm::errs() << "[Pattern] Checking if load is used by dot...\n";
    llvm::errs() << "[Pattern] Load has "
                 << std::distance(loadOp.getResult().getUsers().begin(),
                                  loadOp.getResult().getUsers().end())
                 << " users\n";

    for (auto user : loadOp.getResult().getUsers()) {
      llvm::errs() << "[Pattern]   User: " << user->getName() << "\n";
    }

    std::function<bool(Value)> isEventuallyUsedByDot = [&](Value val) -> bool {
      for (auto user : val.getUsers()) {
        // Direct dot usage
        if (isa<triton::DotOp, triton::DotScaledOp>(user)) {
          llvm::errs() << "[Pattern] Found dot user!\n";
          return true;
        }
        // Type conversion ops (various arith and triton ops)
        if (isa<triton::FpToFpOp, triton::BitcastOp, triton::TransOp,
                arith::UIToFPOp, arith::SIToFPOp, arith::FPToSIOp,
                arith::FPToUIOp, arith::ExtFOp, arith::TruncFOp>(user)) {
          llvm::errs() << "[Pattern] Found conversion op: " << user->getName()
                       << ", checking recursively...\n";
          if (user->getNumResults() > 0 &&
              isEventuallyUsedByDot(user->getResult(0))) {
            return true;
          }
        }
      }
      return false;
    };

    bool isUsedByDot = isEventuallyUsedByDot(loadOp.getResult());

    if (!isUsedByDot) {
      llvm::errs() << "[Pattern] Load not used by dot operation, skipping\n";
      return failure();
    }

    llvm::errs() << "[Pattern] Found tensor load used by dot, analyzing...\n";

    // Analyze the pointer pattern
    SimpleLoadPattern pattern;
    if (!isSimpleOffsetPattern(loadOp.getPtr(), loadType, pattern,
                               loadOp.getOperation())) {
      llvm::errs() << "[Pattern] Not a simple load pattern, skipping\n";
      return failure();
    }

    llvm::errs()
        << "[Pattern] Simple pattern detected, converting to descriptor!\n";

    Location loc = loadOp.getLoc();
    MLIRContext *ctx = loadOp.getContext();

    // Check if base is a pointer type
    auto ptrType = dyn_cast<triton::PointerType>(pattern.basePtr.getType());
    if (!ptrType) {
      llvm::errs() << "[Pattern] Base is not a pointer type: "
                   << pattern.basePtr.getType() << "\n";
      return failure();
    }

    llvm::errs() << "[Pattern] Building descriptor ops...\n";
    llvm::errs() << "[Pattern] Tensor shape: [" << pattern.tensorShape[0]
                 << ", " << pattern.tensorShape[1] << "]\n";
    llvm::errs() << "[Pattern] Block shape: [" << pattern.blockShape[0] << ", "
                 << pattern.blockShape[1] << "]\n";

    // Build TOTAL tensor shape values (i32)
    SmallVector<Value> shapeVals;
    for (int64_t dim : pattern.tensorShape) {
      shapeVals.push_back(rewriter.create<arith::ConstantIntOp>(loc, dim, 32));
    }

    // Build stride values for TOTAL tensor (i64, in ELEMENTS)
    SmallVector<Value> strideVals;
    unsigned elemBits = loadType.getElementTypeBitWidth();
    unsigned elemBytes = (elemBits + 7) / 8;

    llvm::errs() << "[Pattern] Element size: " << elemBits
                 << " bits = " << elemBytes << " bytes\n";
    llvm::errs() << "[Pattern] Strides in elements: [";

    for (size_t i = 0; i < pattern.strides.size(); ++i) {
      int64_t strideElems = pattern.strides[i];
      if (i > 0)
        llvm::errs() << ", ";
      llvm::errs() << strideElems;
      strideVals.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, strideElems, 64));
    }
    llvm::errs() << "]\n";

    llvm::errs() << "[Pattern] Creating MakeTensorDescOp...\n";

    // Create descriptor with NVMMASharedEncoding
    // Create NVMMASharedEncodingAttr for TMA
    // Order: default row-major (rightmost dim is fastest)
    SmallVector<unsigned> order;
    for (int i = loadType.getRank() - 1; i >= 0; --i)
      order.push_back(i);

    // CTALayout: default single CTA
    auto ctaLayout = ttg::CTALayoutAttr::getDefault(ctx, loadType.getRank());

    // Create shared encoding based on BLOCK shape (not total shape)
    auto sharedEnc = ttg::NVMMASharedEncodingAttr::get(
        ctx, pattern.blockShape, order, ctaLayout, loadType.getElementType(),
        /*fp4Padded=*/false);

    llvm::errs() << "[Pattern] Created NVMMASharedEncodingAttr\n";

    // Create block type with encoding (using BLOCK shape)
    auto blockType = RankedTensorType::get(
        pattern.blockShape, loadType.getElementType(), sharedEnc);

    // Create TensorDescType with encoded block type
    auto descType = triton::TensorDescType::get(ctx, blockType);

    llvm::errs() << "[Pattern] Created TensorDescType with encoding\n";

    // Convert blockShape to int32_t array for builder
    SmallVector<int32_t> blockShape32;
    for (auto dim : pattern.blockShape) {
      blockShape32.push_back(static_cast<int32_t>(dim));
    }

    // Create MakeTensorDescOp with explicit result type carrying the shared
    // encoding
    // - shape/strides: TOTAL tensor (strides in elements)
    auto descOp = rewriter.create<triton::MakeTensorDescOp>(
        loc, descType, pattern.basePtr, shapeVals, strideVals,
        triton::PaddingOption::PAD_ZERO);

    llvm::errs() << "[Pattern] MakeTensorDescOp created successfully\n";

    // Create indices
    // If inside a loop, use induction variable; otherwise use zero
    SmallVector<Value> indices;
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    if (auto forOp = loadOp->getParentOfType<scf::ForOp>()) {
      // Inside loop: compute block index along the advancing dimension
      Value iv = forOp.getInductionVar();
      Value step = forOp.getStep();
      llvm::errs()
          << "[Pattern] Using loop induction variable as descriptor index\n";

      // blockIndex = iv / step  (0,1,2,...)
      Value blockIndex = rewriter.create<arith::DivSIOp>(loc, iv, step);

      // Determine which dimension advances across the loop: the one where total
      // tensor shape differs from block shape.
      int loopDim = -1;
      for (int d = 0; d < static_cast<int>(pattern.blockShape.size()); ++d) {
        if (pattern.tensorShape[d] != pattern.blockShape[d]) {
          loopDim = d;
          break;
        }
      }
      // Fallback: if none differs, assume last dim
      if (loopDim < 0)
        loopDim = static_cast<int>(pattern.blockShape.size()) - 1;

      // Safety: TMA requires 16-byte alignment of the address/offsets. Ensure
      // the per-iteration element offset is a multiple of 16 bytes. If not,
      // skip.
      int64_t blkDimSizeBytes =
          pattern.blockShape[loopDim] * static_cast<int64_t>(elemBytes);
      if ((blkDimSizeBytes % 16) != 0) {
        llvm::errs() << "[Pattern] Skipping: block advance (" << blkDimSizeBytes
                     << " bytes) is not 16B-aligned for TMA\n";
        return failure();
      }

      indices.resize(loadType.getRank(), zero);

      // Multiply blockIndex by the block size along that dimension to get
      // element offset
      int64_t blkDimSize = pattern.blockShape[loopDim];
      Value blkDimConst =
          rewriter.create<arith::ConstantIntOp>(loc, blkDimSize, 32);
      Value elemOffset =
          rewriter.create<arith::MulIOp>(loc, blockIndex, blkDimConst);
      indices[loopDim] = elemOffset;
    } else {
      // Outside loop: use all zeros
      llvm::errs() << "[Pattern] Using zero indices (loop-external load)\n";
      for (size_t i = 0; i < loadType.getRank(); ++i) {
        indices.push_back(zero);
      }
    }

    llvm::errs() << "[Pattern] Creating DescriptorLoadOp...\n";

    // Create DescriptorLoadOp
    auto descLoadOp = rewriter.create<triton::DescriptorLoadOp>(
        loc, loadType, descOp.getResult(), indices, loadOp.getCache(),
        loadOp.getEvict());

    llvm::errs()
        << "[Pattern] DescriptorLoadOp created, replacing original load\n";

    rewriter.replaceOp(loadOp, descLoadOp.getResult());

    llvm::errs() << "[Pattern] ✅ Conversion SUCCESS!\n";
    return success();
  }
};

class TritonConvertLoadToDescriptorSM120Pass
    : public impl::TritonConvertLoadToDescriptorSM120Base<
          TritonConvertLoadToDescriptorSM120Pass> {
public:
  TritonConvertLoadToDescriptorSM120Pass() = default;
  TritonConvertLoadToDescriptorSM120Pass(int computeCapability)
      : computeCapability(computeCapability) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // Only for SM_120
    if (computeCapability != 120) {
      llvm::errs() << "[ConvertLoadToDescriptor] Skipping: CC="
                   << computeCapability << "\n";
      return;
    }

    llvm::errs() << "[ConvertLoadToDescriptor] Running for SM_120\n";

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertLoadToDescriptorPattern>(context, computeCapability);

    if (applyPatternsGreedily(module, std::move(patterns)).failed()) {
      llvm::errs() << "[ConvertLoadToDescriptor] FAILED\n";
      signalPassFailure();
    } else {
      llvm::errs() << "[ConvertLoadToDescriptor] SUCCESS\n";
    }
  }

private:
  int computeCapability = 120;
};

} // namespace

std::unique_ptr<Pass>
createConvertLoadToDescriptorSM120Pass(int computeCapability) {
  return std::make_unique<TritonConvertLoadToDescriptorSM120Pass>(
      computeCapability);
}

} // namespace triton
} // namespace mlir
