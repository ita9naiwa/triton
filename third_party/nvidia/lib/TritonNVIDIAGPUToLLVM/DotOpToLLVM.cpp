#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;


// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

LogicalResult convertMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                         const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter, bool isTuring,
                         bool isHopperF64);
LogicalResult convertWGMMA(triton::nvidia_gpu::WarpGroupDotOp op,
                           triton::nvidia_gpu::WarpGroupDotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);
namespace {
struct ScaledDotOpConversion : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern<triton::DotScaledOp>::ConvertOpToLLVMPattern;
int computeCapability;
  ScaledDotOpConversion(LLVMTypeConverter &converter, int computeCapability,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotScaledOp>(converter, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "\n=== ScaledDotOpConversion START ===\n";

    // Basic operation info
    llvm::errs() << "Operation: " << op->getName() << "\n";
    llvm::errs() << "Location: " << op->getLoc() << "\n";
    llvm::errs() << "computeCapability: " << computeCapability << "\n";

    // Result type and encoding
    llvm::errs() << "결과 타입: " << op.getType() << "\n";
    llvm::errs() << "결과 인코딩: " << op.getType().getEncoding() << "\n";

    // Operand types (original values)
    llvm::errs() << "\n--- Original Operands ---\n";
    llvm::errs() << "A 타입: " << op.getA().getType() << "\n";
    llvm::errs() << "B 타입: " << op.getB().getType() << "\n";
    llvm::errs() << "C 타입: " << op.getC().getType() << "\n";
    if (op.getAScale()) {
      llvm::errs() << "AScale 타입: " << op.getAScale().getType() << "\n";
    } else {
      llvm::errs() << "AScale: null\n";
    }
    if (op.getBScale()) {
      llvm::errs() << "BScale 타입: " << op.getBScale().getType() << "\n";
    } else {
      llvm::errs() << "BScale: null\n";
    }

    // Adapted operands (converted by type converter)
    llvm::errs() << "\n--- Adapted Operands (LLVM converted) ---\n";
    llvm::errs() << "adaptedA 타입: " << adaptor.getA().getType() << "\n";
    llvm::errs() << "adaptedB 타입: " << adaptor.getB().getType() << "\n";
    llvm::errs() << "adaptedC 타입: " << adaptor.getC().getType() << "\n";
    if (adaptor.getAScale()) {
      llvm::errs() << "adaptedAScale 타입: " << adaptor.getAScale().getType() << "\n";
    } else {
      llvm::errs() << "adaptedAScale: null\n";
    }
    if (adaptor.getBScale()) {
      llvm::errs() << "adaptedBScale 타입: " << adaptor.getBScale().getType() << "\n";
    } else {
      llvm::errs() << "adaptedBScale: null\n";
    }

    // Attributes
    llvm::errs() << "\n--- Attributes ---\n";
    llvm::errs() << "AElemType: " << static_cast<int>(op.getAElemType()) << "\n";
    llvm::errs() << "BElemType: " << static_cast<int>(op.getBElemType()) << "\n";
    llvm::errs() << "FastMath: " << op.getFastMath() << "\n";
    llvm::errs() << "LhsKPack: " << op.getLhsKPack() << "\n";
    llvm::errs() << "RhsKPack: " << op.getRhsKPack() << "\n";

    // All attributes
    llvm::errs() << "\n--- All Attributes ---\n";
    for (const auto& attr : op->getAttrs()) {
      llvm::errs() << "  " << attr.getName() << ": " << attr.getValue() << "\n";
    }

    // Shape information
    llvm::errs() << "\n--- Shape Information ---\n";
    auto AShapePerCTA = getShapePerCTA(op.getA().getType());
    llvm::errs() << "A shape per CTA: [";
    for (size_t i = 0; i < AShapePerCTA.size(); ++i) {
      llvm::errs() << AShapePerCTA[i];
      if (i < AShapePerCTA.size() - 1) llvm::errs() << ", ";
    }
    llvm::errs() << "]\n";

    auto resultShapePerCTA = getShapePerCTA(op.getType());
    llvm::errs() << "Result shape per CTA: [";
    for (size_t i = 0; i < resultShapePerCTA.size(); ++i) {
      llvm::errs() << resultShapePerCTA[i];
      if (i < resultShapePerCTA.size() - 1) llvm::errs() << ", ";
    }
    llvm::errs() << "]\n";

    // Check if MMA encoding
    if (auto mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(
            cast<RankedTensorType>(op.getType()).getEncoding())) {
      llvm::errs() << "\n--- MMA Encoding Info ---\n";
      llvm::errs() << "MMA versionMajor: " << mmaLayout.getVersionMajor() << "\n";
      llvm::errs() << "MMA versionMinor: " << mmaLayout.getVersionMinor() << "\n";
      llvm::errs() << "MMA warpsPerCTA: [";
      auto warps = mmaLayout.getWarpsPerCTA();
      for (size_t i = 0; i < warps.size(); ++i) {
        llvm::errs() << warps[i];
        if (i < warps.size() - 1) llvm::errs() << ", ";
      }
      llvm::errs() << "]\n";
    }

    // Check if blocked encoding
    if (isa<BlockedEncodingAttr>(cast<RankedTensorType>(op.getType()).getEncoding())) {
      llvm::errs() << "\n--- Blocked Encoding detected ---\n";
    }

    llvm::errs() << "\n=== ScaledDotOpConversion END ===\n";
    llvm::errs() << "하하하하 Done - no matching encoding\n";
    return failure();
  }
};

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;
  int computeCapability;
  DotOpConversion(LLVMTypeConverter &converter, int computeCapability,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    NvidiaMmaEncodingAttr mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    if (!isOuter && mmaLayout && supportMMA(op, mmaLayout.getVersionMajor())) {
      if (mmaLayout.getVersionMajor() == 2) {
        bool isHopperF64 =
            computeCapability == 90 &&
            cast<RankedTensorType>(A.getType()).getElementType().isF64();
        return convertMMA(op, adaptor, getTypeConverter(), rewriter,
                          mmaLayout.isTuring(), isHopperF64);
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct WarpGroupDotOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::WarpGroupDotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // D = A * B + C
    Value A = op.getA();
    TypedValue<RankedTensorType> D = op.getResult();

    // Check if this is a scaled WGMMA operation
    if (op->hasAttr("triton.is_scaled")) {
      llvm::errs() << "Converting scaled WarpGroupDotOp\n";

      // Extract scaling information from attributes
      auto aElemTypeAttr = op->getAttr("triton.a_elem_type");
      auto bElemTypeAttr = op->getAttr("triton.b_elem_type");
      auto lhsKPackAttr = op->getAttr("triton.lhs_k_pack");
      auto rhsKPackAttr = op->getAttr("triton.rhs_k_pack");

      if (!aElemTypeAttr || !bElemTypeAttr) {
        return op.emitError("Scaled WarpGroupDotOp missing element type attributes");
      }

      // Get element types from attributes
      int aElemType = cast<IntegerAttr>(aElemTypeAttr).getInt();
      int bElemType = cast<IntegerAttr>(bElemTypeAttr).getInt();
      bool lhsKPack = lhsKPackAttr ? cast<BoolAttr>(lhsKPackAttr).getValue() : true;
      bool rhsKPack = rhsKPackAttr ? cast<BoolAttr>(rhsKPackAttr).getValue() : true;

      llvm::errs() << "Scaled WGMMA: aElemType=" << aElemType << ", bElemType=" << bElemType << "\n";

      // For now, return an error with detailed info - TODO: implement scaled WGMMA
      return op.emitError("Scaled WarpGroupDotOp conversion not yet implemented");
    }

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    auto mmaLayout = cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (!isOuter && supportMMA(op.getOperand(0), mmaLayout.getVersionMajor())) {
      return convertWGMMA(op, adaptor, getTypeConverter(), rewriter,
                          getThreadId(rewriter, loc));
    }

    return op.emitError(
        "Unsupported WarpGroupDotOp found when converting TritonGPU to LLVM.");
  }
};

struct WarpGroupDotWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::WarpGroupDotWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::WarpGroupDotWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::WarpGroupDotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pendings = op.getPendings();
    Location loc = op.getLoc();
    if (adaptor.getInputs().size() <= 1) {
      Value input =
          adaptor.getInputs().size() == 1 ? adaptor.getInputs()[0] : Value();
      rewriter.replaceOpWithNewOp<triton::nvgpu::WGMMAWaitGroupOp>(op, input,
                                                                   pendings);
      return success();
    }
    std::vector<Type> types;
    // Pack the inputs into a single struct.
    for (Value input : adaptor.getInputs()) {
      auto structType = dyn_cast<LLVM::LLVMStructType>(input.getType());
      if (!structType)
        return failure();
      for (Type type : structType.getBody())
        types.push_back(type);
    }
    auto packedType =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
    Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
    unsigned outputStructIndex = 0;
    for (Value input : adaptor.getInputs()) {
      auto structType = dyn_cast<LLVM::LLVMStructType>(input.getType());
      for (unsigned i = 0; i < structType.getBody().size(); ++i) {
        Value value = rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i], input, i);
        packed = rewriter.create<LLVM::InsertValueOp>(
            loc, packedType, packed, value, outputStructIndex++);
      }
    }
    Value packedOutput =
        rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(loc, packed, pendings);
    // Unpack the output into the original struct types.
    SmallVector<Value> outputs;
    outputStructIndex = 0;
    for (Value input : adaptor.getInputs()) {
      auto structType = cast<LLVM::LLVMStructType>(input.getType());
      Value unpacked = rewriter.create<LLVM::UndefOp>(loc, structType);
      for (unsigned i = 0; i < structType.getBody().size(); ++i) {
        Value value = rewriter.create<LLVM::ExtractValueOp>(
            loc, packedType.getBody()[outputStructIndex], packedOutput,
            outputStructIndex);
        outputStructIndex++;
        unpacked = rewriter.create<LLVM::InsertValueOp>(loc, structType,
                                                        unpacked, value, i);
      }
      outputs.push_back(unpacked);
    }
    rewriter.replaceOp(op, outputs);
    return success();
  }
};
} // namespace

void mlir::triton::NVIDIA::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, computeCapability, benefit);
  patterns.add<WarpGroupDotOpConversion>(typeConverter, benefit);
  patterns.add<WarpGroupDotWaitOpConversion>(typeConverter, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, computeCapability, benefit);
}
