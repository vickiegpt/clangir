//====- LowerMLIRToCIR.cpp - Lowering from MLIR to CIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR-lowered MLIR operations to LLVMIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Passes.h"

using namespace cir;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Bitfield Lowering Helpers
//===----------------------------------------------------------------------===//

static mlir::Value getConst(mlir::OpBuilder &bld, mlir::Location loc,
                            mlir::Type typ, unsigned val) {
  return bld.create<mlir::LLVM::ConstantOp>(loc, typ, val);
}

static mlir::Value getConstAPInt(mlir::OpBuilder &bld, mlir::Location loc,
                                 mlir::Type typ, const llvm::APInt &val) {
  return bld.create<mlir::LLVM::ConstantOp>(loc, typ, val);
}

static mlir::Value createShL(mlir::OpBuilder &bld, mlir::Value lhs,
                             unsigned rhs) {
  if (!rhs)
    return lhs;
  auto rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::ShlOp>(lhs.getLoc(), lhs, rhsVal);
}

static mlir::Value createLShR(mlir::OpBuilder &bld, mlir::Value lhs,
                              unsigned rhs) {
  if (!rhs)
    return lhs;
  auto rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::LShrOp>(lhs.getLoc(), lhs, rhsVal);
}

static mlir::Value createAShR(mlir::OpBuilder &bld, mlir::Value lhs,
                              unsigned rhs) {
  if (!rhs)
    return lhs;
  auto rhsVal = getConst(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::AShrOp>(lhs.getLoc(), lhs, rhsVal);
}

static mlir::Value createAnd(mlir::OpBuilder &bld, mlir::Value lhs,
                             const llvm::APInt &rhs) {
  auto rhsVal = getConstAPInt(bld, lhs.getLoc(), lhs.getType(), rhs);
  return bld.create<mlir::LLVM::AndOp>(lhs.getLoc(), lhs, rhsVal);
}

static mlir::Value createIntCast(mlir::OpBuilder &bld, mlir::Value src,
                                 mlir::IntegerType dstTy,
                                 bool isSigned = false) {
  auto srcTy = src.getType();
  assert(mlir::isa<mlir::IntegerType>(srcTy));

  auto srcWidth = mlir::cast<mlir::IntegerType>(srcTy).getWidth();
  auto dstWidth = mlir::cast<mlir::IntegerType>(dstTy).getWidth();
  auto loc = src.getLoc();

  if (dstWidth > srcWidth && isSigned)
    return bld.create<mlir::LLVM::SExtOp>(loc, dstTy, src);
  else if (dstWidth > srcWidth)
    return bld.create<mlir::LLVM::ZExtOp>(loc, dstTy, src);
  else if (dstWidth < srcWidth)
    return bld.create<mlir::LLVM::TruncOp>(loc, dstTy, src);
  else
    return bld.create<mlir::LLVM::BitcastOp>(loc, dstTy, src);
}

//===----------------------------------------------------------------------===//
// Bitfield Op Lowering Patterns
//===----------------------------------------------------------------------===//

class CIRGetBitfieldOpLowering
    : public mlir::OpConversionPattern<cir::GetBitfieldOp> {
public:
  using mlir::OpConversionPattern<cir::GetBitfieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetBitfieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto info = op.getBitfieldInfo();
    auto size = info.getSize();
    auto offset = info.getOffset();
    auto storageType = info.getStorageType();
    auto context = storageType.getContext();
    unsigned storageSize = 0;

    if (auto arTy = mlir::dyn_cast<cir::ArrayType>(storageType))
      storageSize = arTy.getSize() * 8;
    else if (auto intTy = mlir::dyn_cast<cir::IntType>(storageType))
      storageSize = intTy.getWidth();
    else
      return rewriter.notifyMatchFailure(
          op, "Expected ArrayType or IntType for bitfield storage");

    auto intType = mlir::IntegerType::get(context, storageSize);

    mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), intType, adaptor.getAddr(), op.getAlignment(),
        op.getIsVolatile());
    val = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), intType, val);

    if (info.getIsSigned()) {
      assert(static_cast<unsigned>(offset + size) <= storageSize);
      unsigned highBits = storageSize - offset - size;
      val = createShL(rewriter, val, highBits);
      val = createAShR(rewriter, val, offset + highBits);
    } else {
      val = createLShR(rewriter, val, offset);

      if (static_cast<unsigned>(offset) + size < storageSize)
        val = createAnd(rewriter, val,
                        llvm::APInt::getLowBitsSet(storageSize, size));
    }

    // Convert the result type - the CIR type is IntType which needs to become
    // an LLVM integer type
    auto resultCirType = op.getType();
    unsigned resultWidth = 0;
    bool resultSigned = false;
    if (auto cirIntType = mlir::dyn_cast<cir::IntType>(resultCirType)) {
      resultWidth = cirIntType.getWidth();
      resultSigned = cirIntType.isSigned();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Expected IntType for bitfield result");
    }
    auto resTy = mlir::IntegerType::get(context, resultWidth);
    auto newOp = createIntCast(rewriter, val, resTy, resultSigned);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

class CIRSetBitfieldOpLowering
    : public mlir::OpConversionPattern<cir::SetBitfieldOp> {
public:
  using mlir::OpConversionPattern<cir::SetBitfieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SetBitfieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto info = op.getBitfieldInfo();
    auto size = info.getSize();
    auto offset = info.getOffset();
    auto storageType = info.getStorageType();
    auto context = storageType.getContext();

    unsigned storageSize = 0;

    if (auto arTy = mlir::dyn_cast<cir::ArrayType>(storageType))
      storageSize = arTy.getSize() * 8;
    else if (auto intTy = mlir::dyn_cast<cir::IntType>(storageType))
      storageSize = intTy.getWidth();
    else
      return rewriter.notifyMatchFailure(
          op, "Expected ArrayType or IntType for bitfield storage");

    auto intType = mlir::IntegerType::get(context, storageSize);

    // Convert source value to storage type width
    mlir::Value srcVal = adaptor.getSrc();
    // The source might be an LLVM integer from a previous conversion, or a CIR
    // type
    if (!mlir::isa<mlir::IntegerType>(srcVal.getType())) {
      // Try to convert from CIR type
      if (auto cirIntType = mlir::dyn_cast<cir::IntType>(srcVal.getType())) {
        auto srcWidth = cirIntType.getWidth();
        auto srcLLVMType = mlir::IntegerType::get(context, srcWidth);
        srcVal = rewriter.create<mlir::UnrealizedConversionCastOp>(
                              op.getLoc(), srcLLVMType, srcVal)
                     .getResult(0);
      }
    }
    srcVal = createIntCast(rewriter, srcVal, intType, info.getIsSigned());
    auto srcWidth = storageSize;
    auto resultVal = srcVal;

    if (storageSize != size) {
      assert(storageSize > size && "Invalid bitfield size.");

      mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
          op.getLoc(), intType, adaptor.getAddr(), op.getAlignment(),
          op.getIsVolatile());

      srcVal = createAnd(rewriter, srcVal,
                         llvm::APInt::getLowBitsSet(srcWidth, size));
      resultVal = srcVal;
      srcVal = createShL(rewriter, srcVal, offset);

      // Mask out the original value.
      val = createAnd(rewriter, val,
                      ~llvm::APInt::getBitsSet(srcWidth, offset, offset + size));

      // Or together the unchanged values and the source value.
      srcVal = rewriter.create<mlir::LLVM::OrOp>(op.getLoc(), val, srcVal);
    }

    rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), srcVal, adaptor.getAddr(),
                                         op.getAlignment(), op.getIsVolatile());

    // Convert the result type
    auto resultCirType = op.getType();
    unsigned resultWidth = 0;
    bool resultSigned = info.getIsSigned();
    if (auto cirIntType = mlir::dyn_cast<cir::IntType>(resultCirType)) {
      resultWidth = cirIntType.getWidth();
      resultSigned = cirIntType.isSigned();
    } else {
      return rewriter.notifyMatchFailure(
          op, "Expected IntType for bitfield result");
    }
    auto resultTy = mlir::IntegerType::get(context, resultWidth);

    if (info.getIsSigned()) {
      assert(size <= storageSize);
      unsigned highBits = storageSize - size;

      if (highBits) {
        resultVal = createShL(rewriter, resultVal, highBits);
        resultVal = createAShR(rewriter, resultVal, highBits);
      }
    }

    resultVal = createIntCast(rewriter, resultVal, resultTy, resultSigned);

    rewriter.replaceOp(op, resultVal);
    return mlir::success();
  }
};

namespace cir {
struct ConvertMLIRToLLVMPass
    : public mlir::PassWrapper<ConvertMLIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::cf::ControlFlowDialect>();
  }
  void runOnOperation() final;

  StringRef getDescription() const override {
    return "Convert the MLIR standard dialects produced from CIR to MLIR LLVM "
           "dialect";
  }

  StringRef getArgument() const override { return "cir-mlir-to-llvm"; }
};

void ConvertMLIRToLLVMPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // Add CIR bitfield lowering patterns
  patterns.add<CIRGetBitfieldOpLowering, CIRSetBitfieldOpLowering>(
      typeConverter, &getContext());
  target.addIllegalOp<cir::GetBitfieldOp, cir::SetBitfieldOp>();

  auto module = getOperation();

  // Lower the module attributes to LLVM equivalents.
  if (auto tripleAttr = module->getAttr(cir::CIRDialect::getTripleAttrName()))
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    tripleAttr);

  // Strip the CIR attributes.
  module->removeAttr(cir::CIRDialect::getSOBAttrName());
  module->removeAttr(cir::CIRDialect::getSourceLanguageAttrName());
  module->removeAttr(cir::CIRDialect::getTripleAttrName());

  // Pre-legalization fixup: ensure every memref.alloca_scope region block is
  // properly terminated with memref.alloca_scope.return. Some upstream
  // transformations may leave blocks without the expected terminator,
  // which causes the LLVM conversion to crash when converting the op.
  module.walk([&](mlir::Operation *op) {
    if (auto scope = llvm::dyn_cast<mlir::memref::AllocaScopeOp>(op)) {
      if (!scope->getNumRegions())
        return;
      auto &region = scope.getRegion();
      if (region.empty())
        return;
      // Check ALL blocks in the region, not just the first one
      for (auto &block : region) {
        if (block.empty() || !block.getTerminator() ||
            !llvm::isa<mlir::memref::AllocaScopeReturnOp>(block.getTerminator())) {
          mlir::OpBuilder builder(&block, block.end());
          builder.create<mlir::memref::AllocaScopeReturnOp>(scope.getLoc());
        }
      }
    }
  });

  // Lower llvm.intr.is.constant to constant false.
  // At runtime, llvm.is.constant should return false for non-constant values.
  // There's a bug in LLVM 22's codegen with this intrinsic, so we lower it
  // early to avoid the crash.
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "llvm.intr.is.constant") {
      mlir::OpBuilder builder(op);
      auto falseVal = builder.create<mlir::LLVM::ConstantOp>(
          op->getLoc(), builder.getI1Type(), builder.getBoolAttr(false));
      op->getResult(0).replaceAllUsesWith(falseVal);
      op->erase();
    }
  });

  // Remove orphaned blocks that contain CIR operations. These are cleanup
  // blocks that became unreachable during the CIR-to-MLIR lowering. They
  // contain destructor calls that can't be properly integrated without full
  // exception handling support.
  module.walk([&](mlir::func::FuncOp func) {
    llvm::SmallVector<mlir::Block *, 8> blocksToRemove;
    for (auto &block : func.getBody()) {
      // Skip entry block
      if (&block == &func.getBody().front())
        continue;
      // Check if block has no predecessors (orphaned)
      if (block.hasNoPredecessors()) {
        // Check if block contains CIR operations
        bool hasCIROps = false;
        for (auto &op : block) {
          if (op.getDialect() &&
              op.getDialect()->getNamespace() == "cir") {
            hasCIROps = true;
            break;
          }
        }
        if (hasCIROps) {
          blocksToRemove.push_back(&block);
        }
      }
    }
    // Erase orphaned blocks (in reverse to maintain iterator validity)
    for (auto *block : llvm::reverse(blocksToRemove)) {
      block->dropAllDefinedValueUses();
      block->erase();
    }
  });

  // Also remove orphaned blocks in LLVM functions
  module.walk([&](mlir::LLVM::LLVMFuncOp func) {
    llvm::SmallVector<mlir::Block *, 8> blocksToRemove;
    for (auto &block : func.getBody()) {
      // Skip entry block
      if (&block == &func.getBody().front())
        continue;
      // Check if block has no predecessors (orphaned)
      if (block.hasNoPredecessors()) {
        // Check if block contains CIR operations
        bool hasCIROps = false;
        for (auto &op : block) {
          if (op.getDialect() &&
              op.getDialect()->getNamespace() == "cir") {
            hasCIROps = true;
            break;
          }
        }
        if (hasCIROps) {
          blocksToRemove.push_back(&block);
        }
      }
    }
    // Erase orphaned blocks (in reverse to maintain iterator validity)
    for (auto *block : llvm::reverse(blocksToRemove)) {
      block->dropAllDefinedValueUses();
      block->erase();
    }
  });

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();

  // Post-processing: Remove duplicate terminators in basic blocks.
  // C++ exception handling can produce patterns where llvm.unreachable follows
  // llvm.return (or other terminators) in the same block. This is invalid LLVM IR.
  // Walk through all LLVM functions and remove any llvm.unreachable that follows
  // another terminator.
  module.walk([&](mlir::LLVM::LLVMFuncOp func) {
    llvm::SmallVector<mlir::Operation *, 16> opsToErase;
    for (auto &block : func.getBody()) {
      bool foundTerminator = false;
      for (auto &op : block) {
        if (foundTerminator) {
          // Everything after the first terminator should be removed
          opsToErase.push_back(&op);
        } else if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
          foundTerminator = true;
        }
      }
    }
    for (auto *op : opsToErase) {
      op->dropAllUses();
      op->erase();
    }
  });

  // Reconcile any remaining unrealized_conversion_cast operations.
  // These may arise from intermediate type conversions during the lowering.
  llvm::SmallVector<mlir::UnrealizedConversionCastOp, 16> castOps;
  module.walk([&](mlir::UnrealizedConversionCastOp op) {
    castOps.push_back(op);
  });
  mlir::reconcileUnrealizedCasts(castOps);
}

std::unique_ptr<mlir::Pass> createConvertMLIRToLLVMPass() {
  return std::make_unique<ConvertMLIRToLLVMPass>();
}

} // namespace cir
