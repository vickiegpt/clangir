//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/Passes.h"
#include "clang/CIR/Passes.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace mlir;

namespace mlir {
namespace {
static LogicalResult sanitizeCIRGotoInLLVMFuncs(ModuleOp module) {
  module.walk([&](LLVM::LLVMFuncOp funcOp) {
    llvm::StringMap<Block *> labels;
    SmallVector<Operation *> gotos;

    funcOp.walk([&](Operation *nested) {
      auto name = nested->getName().getStringRef();
      if (name == "cir.label") {
        if (auto labelAttr =
                nested->getAttrOfType<StringAttr>("label"))
          labels.try_emplace(labelAttr.getValue(), nested->getBlock());
      } else if (name == "cir.goto") {
        gotos.push_back(nested);
      }
    });

    for (Operation *goTo : gotos) {
      auto labelAttr = goTo->getAttrOfType<StringAttr>("label");
      if (!labelAttr)
        continue;

      auto it = labels.find(labelAttr.getValue());
      if (it == labels.end()) {
        goTo->emitRemark()
            << "unresolved goto label '" << labelAttr.getValue()
            << "'; replacing with llvm.unreachable";
        OpBuilder builder(goTo);
        builder.create<LLVM::UnreachableOp>(goTo->getLoc());
        goTo->erase();
        continue;
      }

      OpBuilder builder(goTo);
      builder.create<LLVM::BrOp>(goTo->getLoc(), it->second);
      goTo->erase();
    }
  });

  return success();
}
/// Return true if the module contains any CIR constructs (ops, types, or
/// attributes). We use this to decide whether to run CIR-specific lowerings.
static bool containsCIR(ModuleOp module) {
  bool found = false;
  // Quick check for module-level CIR attributes.
  for (auto namedAttr : module->getAttrs()) {
    if (namedAttr.getName().strref().starts_with("cir."))
      return true;
  }
  module.walk([&](Operation *op) {
    // CIR operation by name.
    if (op->getName().getStringRef().starts_with("cir.")) {
      found = true;
      return WalkResult::interrupt();
    }
    // CIR-typed results or operands.
    auto hasCIRType = [](Type t) {
      if (!t)
        return false;
      Dialect &dialect = t.getDialect();
      if (dialect.getNamespace() == "cir")
        return true;
      return false;
    };
    for (Type ty : op->getResultTypes())
      if (hasCIRType(ty)) {
        found = true;
        return WalkResult::interrupt();
      }
    for (Value v : op->getOperands())
      if (hasCIRType(v.getType())) {
        found = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return found;
}

/// Try to remove leftover pointer bridge casts between LLVM and CIR types.
/// Pattern: %c = builtin.unrealized_conversion_cast %p : !llvm.ptr -> !cir.ptr<...>
///          %q = builtin.unrealized_conversion_cast %c : !cir.ptr<...> -> !llvm.ptr
/// We rewrite uses of %q to %p and erase the bridge casts if possible.
static void cleanupPointerBridgeCasts(ModuleOp module) {
  SmallVector<Operation *> toErase;
  auto isCIRType = [](Type t) {
    if (!t)
      return false;
    Dialect &d = t.getDialect();
    return d.getNamespace() == "cir";
  };

  module.walk([&](UnrealizedConversionCastOp uc1) {
    // Only handle simple 1->1 bridge casts.
    if (uc1.getInputs().size() != 1 || uc1.getOutputs().size() != 1)
      return WalkResult::advance();

    Value in = uc1.getInputs().front();
    Value out = uc1.getOutputs().front();
    auto inTy = in.getType();
    auto outTy = out.getType();
    // Look for llvm.ptr -> cir.ptr bridges.
    if (!mlir::isa_and_nonnull<LLVM::LLVMPointerType>(inTy) || !isCIRType(outTy))
      return WalkResult::advance();

    // Try to collapse subsequent cir.ptr -> llvm.* bridges that only feed
    // back into LLVM-typed values.
    SmallVector<UnrealizedConversionCastOp> secondaries;
    for (OpOperand &use : llvm::make_early_inc_range(out.getUses())) {
      if (auto uc2 = dyn_cast<UnrealizedConversionCastOp>(use.getOwner())) {
        // Must use 'out' as one of the inputs.
        bool usesOut = llvm::any_of(uc2.getInputs(), [&](Value v) { return v == out; });
        if (!usesOut)
          continue;
        // Only handle simple 1->1 back to LLVM types.
        if (uc2.getInputs().size() != 1 || uc2.getOutputs().size() != 1)
          continue;
        Type resTy = uc2.getOutputs().front().getType();
        Dialect &resDialect = resTy.getDialect();
        if (resDialect.getNamespace() != "llvm")
          continue;
        // Replace the result with original llvm pointer 'in'. Types must match
        // or be another llvm ptr; if not identical, let later casts reconcile.
        uc2.getResult(0).replaceAllUsesWith(in);
        secondaries.push_back(uc2);
      }
    }
    // If the cir.ptr has no more uses, schedule both bridge ops for erasure.
    if (out.use_empty()) {
      toErase.push_back(uc1.getOperation());
      for (auto uc2 : secondaries)
        toErase.push_back(uc2.getOperation());
    }
    return WalkResult::advance();
  });

  for (Operation *op : toErase)
    op->erase();
}

/// Collapse simple round-trip casts bridging builtin integers and CIR integer
/// types, and drop dead CIR-typed unrealized casts.
/// Examples handled:
///   1) %c = builtin.unrealized_conversion_cast %x : i64 -> !cir.int<s,64>
///        %y = builtin.unrealized_conversion_cast %c : !cir.int<s,64> -> i64
///      Replaces `%y` with `%x` and erases both casts if `%c` becomes dead.
///   2) Any builtin.unrealized_conversion_cast whose sole result has a CIR
///      type and has no uses will be erased.
static void cleanupCIRIntBridgeCasts(ModuleOp module) {
  SmallVector<Operation *> toErase;

  auto isCIRType = [](Type t) {
    if (!t)
      return false;
    return t.getDialect().getNamespace() == "cir";
  };

  module.walk([&](UnrealizedConversionCastOp uc1) {
    // Only handle simple 1->1 casts for now.
    if (uc1.getInputs().size() != 1 || uc1.getOutputs().size() != 1)
      return WalkResult::advance();

    Value in = uc1.getInputs().front();
    Value out = uc1.getOutputs().front();
    Type inTy = in.getType();
    Type outTy = out.getType();

    // If the cast produces a CIR type and is dead, erase it.
    if (isCIRType(outTy) && out.use_empty()) {
      toErase.push_back(uc1.getOperation());
      return WalkResult::advance();
    }

    // Look for builtin int -> CIR int -> builtin int pattern.
    if (!isCIRType(outTy))
      return WalkResult::advance();

    if (!mlir::isa<IntegerType>(inTy))
      return WalkResult::advance();

    SmallVector<UnrealizedConversionCastOp> secondaries;
    for (OpOperand &use : llvm::make_early_inc_range(out.getUses())) {
      if (auto uc2 = dyn_cast<UnrealizedConversionCastOp>(use.getOwner())) {
        // Must be a simple 1->1 back conversion using 'out' as the input.
        if (uc2.getInputs().size() != 1 || uc2.getOutputs().size() != 1)
          continue;
        bool usesOut = (uc2.getInputs().front() == out);
        if (!usesOut)
          continue;
        // Only fold back to builtin integers (not CIR types).
        Type resTy = uc2.getOutputs().front().getType();
        if (isCIRType(resTy))
          continue;
        if (!mlir::isa<IntegerType>(resTy))
          continue;

        // Replace the bridged value with the original builtin integer.
        uc2.getResult(0).replaceAllUsesWith(in);
        secondaries.push_back(uc2);
      }
    }

    // If the CIR-typed result no longer has uses, schedule for erasure.
    if (out.use_empty()) {
      toErase.push_back(uc1.getOperation());
      for (auto uc2 : secondaries)
        toErase.push_back(uc2.getOperation());
    }
    return WalkResult::advance();
  });

  for (Operation *op : toErase)
    op->erase();
}
} // namespace

void registerToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](Operation *op, raw_ostream &output) {
        auto module = dyn_cast<ModuleOp>(op);
        if (!module)
          return failure();

        if (failed(sanitizeCIRGotoInLLVMFuncs(module)))
          return failure();

        // Only run CIR lowerings if the module still contains CIR constructs.
        if (containsCIR(module)) {
          PassManager pm(op->getContext());
          // Pre-lowering cleanups to simplify CIR CFG and remove gotos.
          mlir::populateCIRPreLoweringPasses(pm, /*useCCLowering=*/false);
          // Lower CIR to MLIR std dialects and then to LLVM dialect.
          pm.addPass(cir::createConvertCIRToMLIRPass());
          pm.addPass(cir::createConvertMLIRToLLVMPass());
          // Clean up any remaining unrealized casts.
          pm.addPass(createReconcileUnrealizedCastsPass());
          if (failed(pm.run(module.getOperation())))
            return failure();
          // Drop residual CIR<->LLVM bridge casts if any.
          cleanupPointerBridgeCasts(module);
          cleanupCIRIntBridgeCasts(module);
        }

        {
          PassManager pm(op->getContext());
          pm.addPass(createReconcileUnrealizedCastsPass());
          if (failed(pm.run(module.getOperation())))
            return failure();
        }

        // Final best-effort cleanup in case any unrealized casts bridging CIR
        // types are still present but dead after the last reconciliation.
        cleanupPointerBridgeCasts(module);
        cleanupCIRIntBridgeCasts(module);

        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->removeDebugIntrinsicDeclarations();
        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, DLTIDialect, func::FuncDialect,
                        memref::MemRefDialect, scf::SCFDialect, cir::CIRDialect>();
        registerAllToLLVMIRTranslations(registry);
      });
}
} // namespace mlir
