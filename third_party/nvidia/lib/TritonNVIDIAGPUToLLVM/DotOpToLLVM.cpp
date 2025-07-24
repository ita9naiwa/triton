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

// Scaled MMA instruction mapping
inline static const std::map<int, std::string> mmaInstrPtxScaled = {
  { 0, // FP32_FP8E5M2_FP8E5M2_FP32_1X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e5m2.e5m2.f32.ue8m0" },
  { 1, // FP32_FP8E5M2_FP8E4M3FN_FP32_1X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e5m2.e4m3.f32.ue8m0" },
  { 2, // FP32_FP8E4M3FN_FP8E5M2_FP32_1X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0" },
  { 3, // FP32_FP8E4M3FN_FP8E4M3FN_FP32_1X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0" },
  { 4, // FP32_FP8E5M2_FP8E5M2_FP32_2X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e5m2.e5m2.f32.ue8m0" },
  { 5, // FP32_FP8E5M2_FP8E4M3FN_FP32_2X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e5m2.e4m3.f32.ue8m0" },
  { 6, // FP32_FP8E4M3FN_FP8E5M2_FP32_2X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e4m3.e5m2.f32.ue8m0" },
  { 7, // FP32_FP8E4M3FN_FP8E4M3FN_FP32_2X
    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e4m3.e4m3.f32.ue8m0" }
};

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
    llvm::errs() << "하하하하 ScaledDotOpConversion START\n";
    llvm::errs() << "결과 타입: " << op.getType() << "\n";
    llvm::errs() << "결과 인코딩: " << op.getType().getEncoding() << "\n";

    Location loc = op->getLoc();
    Value A = op.getA();
    Value D = op.getResult();

    // Check if we have the required scales
    if (!op.getAScale() || !op.getBScale()) {
      return op.emitError("ScaledDotOp requires both A and B scales");
    }

    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    NvidiaMmaEncodingAttr mmaLayout = dyn_cast<NvidiaMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());

        if (!isOuter && mmaLayout && supportMMA(op.getA(), mmaLayout.getVersionMajor())) {
      if (mmaLayout.getVersionMajor() == 5 && computeCapability >= 100) {
        // TODO: Implement scaled WGMMA conversion for DotScaledOp
        llvm::errs() << "MMA v5 scaled conversion not yet implemented\n";
        return failure();
      } else if (mmaLayout.getVersionMajor() == 2) {
        // TODO: Implement scaled MMA conversion for DotScaledOp
        llvm::errs() << "MMA v2 scaled conversion not yet implemented\n";
        return failure();
      }

      return op.emitError("Unsupported MMA version for ScaledDotOp");
    }

    if (isa<BlockedEncodingAttr>(cast<RankedTensorType>(D.getType()).getEncoding())) {
      // Convert blocked encoding scaled dot to scaled MMA instructions
      llvm::errs() << "Converting blocked ScaledDot to scaled MMA instructions\n";

      // For blocked encoding, we need to use FMA-based approach with scaling
      // This will generate mma.sync.aligned.m16n8k64.row.col.kind.block_scale or
      // mxf8f6f4.block_scale.scale_vec instructions


      if (!adaptor.getAScale() || !adaptor.getBScale()) {
        return op.emitError("ScaledDotOp requires both A and B scales for blocked encoding");
      }

      // Convert operands to appropriate layouts for scaled MMA
      Value adaptedA = adaptor.getA();
      Value adaptedB = adaptor.getB();
      Value adaptedC = adaptor.getC();
      Value adaptedAScale = adaptor.getAScale();
      Value adaptedBScale = adaptor.getBScale();

      // Create the scaled MMA operation using inline assembly or intrinsics
      // This should generate the appropriate PTX instructions
            llvm::errs() << "A 타입: " << adaptedA.getType() << "\n";
      llvm::errs() << "B 타입: " << adaptedB.getType() << "\n";
      llvm::errs() << "A scale 타입: " << adaptedAScale.getType() << "\n";
      llvm::errs() << "B scale 타입: " << adaptedBScale.getType() << "\n";

      // Determine the appropriate scaled MMA instruction
      auto aElemType = cast<RankedTensorType>(A.getType()).getElementType();
      auto bElemType = cast<RankedTensorType>(op.getB().getType()).getElementType();
      auto dElemType = cast<RankedTensorType>(D.getType()).getElementType();

      int instructionKey = -1;
      bool is1X = true; // For now, default to 1X scaling

      // Determine instruction based on A and B element types
      if (isa<Float8E5M2Type>(aElemType) && isa<Float8E5M2Type>(bElemType) && dElemType.isF32()) {
        instructionKey = is1X ? 0 : 4; // FP32_FP8E5M2_FP8E5M2_FP32_1X/2X
      } else if (isa<Float8E5M2Type>(aElemType) && isa<Float8E4M3FNType>(bElemType) && dElemType.isF32()) {
        instructionKey = is1X ? 1 : 5; // FP32_FP8E5M2_FP8E4M3FN_FP32_1X/2X
      } else if (isa<Float8E4M3FNType>(aElemType) && isa<Float8E5M2Type>(bElemType) && dElemType.isF32()) {
        instructionKey = is1X ? 2 : 6; // FP32_FP8E4M3FN_FP8E5M2_FP32_1X/2X
      } else if (isa<Float8E4M3FNType>(aElemType) && isa<Float8E4M3FNType>(bElemType) && dElemType.isF32()) {
        instructionKey = is1X ? 3 : 7; // FP32_FP8E4M3FN_FP8E4M3FN_FP32_1X/2X
      }

      if (instructionKey == -1) {
        return op.emitError("Unsupported element type combination for scaled MMA");
      }

      std::string ptxInstruction = mmaInstrPtxScaled.at(instructionKey);
      llvm::errs() << "Selected PTX: " << ptxInstruction << "\n";

      // Following Triton's ElementwiseInlineAsmOp pattern
      // Pack operands and prepare for inline assembly
      SmallVector<Value> packedOperands = {adaptedA, adaptedB, adaptedC, adaptedAScale, adaptedBScale};

      // For MMA, we typically need multiple outputs (4 for m16n8k64)
      auto i32Ty = rewriter.getI32Type();
      SmallVector<Type> asmRetTypes = {i32Ty, i32Ty, i32Ty, i32Ty}; // 4 outputs

      // Create struct type for multiple results
      Type asmRetType = asmRetTypes.size() > 1 ?
          LLVM::LLVMStructType::getLiteral(rewriter.getContext(), asmRetTypes) :
          asmRetTypes[0];

      // Constraints: 4 outputs + 5 inputs
      std::string constraints = "=r,=r,=r,=r,r,r,r,r,r";

      // Create inline assembly string using the selected PTX instruction
      std::string asmString = ptxInstruction + " "
        "{%0, %1, %2, %3}, "        // 4 output registers
        "{%4}, {%5}, {%6}, "        // A, B, C operands (simplified for now)
        "%7, {0, 1}, %8, {0, 1};";  // scaleA + params, scaleB + params

      llvm::errs() << "Generated ASM: " << asmString << "\n";

      // Create the inline assembly operation
      auto asmResults = rewriter.create<LLVM::InlineAsmOp>(
          loc, asmRetType,
          /*operands=*/packedOperands,
          /*asm_string=*/rewriter.getStringAttr(asmString),
          /*constraints=*/rewriter.getStringAttr(constraints),
          /*has_side_effects=*/false,
          /*is_align_stack=*/false,
          /*tail_call_kind=*/LLVM::tailcallkind::TailCallKind::None,
          /*asm_dialect=*/LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                    LLVM::AsmDialect::AD_ATT),
          /*operand_attrs=*/ArrayAttr())->getResult(0);

            // Extract all results from the struct and reconstruct tensor
      // MMA returns 4 scalar values that need to be packed back into a tensor
      SmallVector<Value> scalarResults;
      if (asmRetTypes.size() > 1) {
        // Extract each scalar result from the struct
        for (unsigned i = 0; i < asmRetTypes.size(); i++) {
          scalarResults.push_back(
            rewriter.create<LLVM::ExtractValueOp>(loc, asmResults, i));
        }
      } else {
        scalarResults.push_back(asmResults);
      }

      // Create a struct to hold all the scalar results
      auto origType = getTypeConverter()->convertType(D.getType());

      // For now, use the struct of scalars as the result
      // TODO: Properly convert back to the original tensor layout
      // This would require understanding the exact memory layout
      Value finalResult = asmResults; // Use the struct directly

      rewriter.replaceOp(op, finalResult);
      llvm::errs() << "result: " << finalResult << "\n";
      llvm::errs() << "하하하하 Done - Generated scaled MMA instruction\n";
      return success();
    }

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
