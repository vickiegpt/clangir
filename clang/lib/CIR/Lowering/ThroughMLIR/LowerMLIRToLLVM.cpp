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
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Passes.h"

using namespace cir;
using namespace llvm;

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
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

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
}

std::unique_ptr<mlir::Pass> createConvertMLIRToLLVMPass() {
  return std::make_unique<ConvertMLIRToLLVMPass>();
}

} // namespace cir
