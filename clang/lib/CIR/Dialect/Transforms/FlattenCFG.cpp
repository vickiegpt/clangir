//====- FlattenCFG.cpp - Flatten CIR CFG ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass that inlines CIR operations regions into the parent
// function region.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

namespace {

/// Lowers operations with the terminator trait that have a single successor.
void lowerTerminator(mlir::Operation *op, mlir::Block *dest,
                     mlir::PatternRewriter &rewriter) {
  assert(op->hasTrait<mlir::OpTrait::IsTerminator>() && "not a terminator");
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<cir::BrOp>(op, dest);
}

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(
    mlir::Region &region,
    mlir::function_ref<mlir::WalkResult(mlir::Operation *)> callback) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<Ops...>(op))
      return mlir::WalkResult::skip();
    return callback(op);
  });
}

struct FlattenCFGPass : public FlattenCFGBase<FlattenCFGPass> {

  FlattenCFGPass() = default;
  void runOnOperation() override;
};

struct CIRIfFlattening : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cir::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = ifOp.getLoc();
    auto emptyElse = ifOp.getElseRegion().empty();

    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (ifOp->getResults().empty())
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region
    auto *thenBeforeBody = &ifOp.getThenRegion().front();
    auto *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<cir::YieldOp>(thenAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<cir::BrOp>(thenYieldOp, thenYieldOp.getArgs(),
                                             continueBlock);
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    mlir::Block *elseBeforeBody = nullptr;
    mlir::Block *elseAfterBody = nullptr;
    if (!emptyElse) {
      elseBeforeBody = &ifOp.getElseRegion().front();
      elseAfterBody = &ifOp.getElseRegion().back();
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), continueBlock);
    } else {
      elseBeforeBody = elseAfterBody = continueBlock;
    }

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<cir::BrCondOp>(loc, ifOp.getCondition(), thenBeforeBody,
                                   elseBeforeBody);

    if (!emptyElse) {
      rewriter.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOp =
              dyn_cast<cir::YieldOp>(elseAfterBody->getTerminator())) {
        rewriter.replaceOpWithNewOp<cir::BrOp>(
            elseYieldOp, elseYieldOp.getArgs(), continueBlock);
      }
    }

    rewriter.replaceOp(ifOp, continueBlock->getArguments());
    return mlir::success();
  }
};

class CIRScopeOpFlattening : public mlir::OpRewritePattern<cir::ScopeOp> {
public:
  using OpRewritePattern<cir::ScopeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ScopeOp scopeOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = scopeOp.getLoc();

    // Empty scope: just remove it.
    // TODO: Remove this logic once CIR uses MLIR infrastructure to remove
    // trivially dead operations
    if (scopeOp.isEmpty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Split the current block before the ScopeOp to create the inlining
    // point.
    auto *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *continueBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

    // Check if we need block arguments for scope results.
    // For C++ materialized temporaries, the scope may have a result type but
    // the yield inside doesn't provide arguments - the result is the address
    // of a local alloca that was defined before the scope.
    auto *afterBody = &scopeOp.getScopeRegion().back();
    auto yieldOp = dyn_cast<cir::YieldOp>(afterBody->getTerminator());
    bool useBlockArgs = scopeOp.getNumResults() > 0 && yieldOp &&
                        yieldOp.getArgs().size() == scopeOp.getNumResults();

    if (useBlockArgs)
      continueBlock->addArguments(scopeOp.getResultTypes(), loc);

    // Inline body region.
    auto *beforeBody = &scopeOp.getScopeRegion().front();
    rewriter.inlineRegionBefore(scopeOp.getScopeRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    // TODO(CIR): stackSaveOp
    // auto stackSaveOp = rewriter.create<mlir::LLVM::StackSaveOp>(
    //     loc, mlir::LLVM::LLVMPointerType::get(
    //              mlir::IntegerType::get(scopeOp.getContext(), 8)));
    rewriter.create<cir::BrOp>(loc, mlir::ValueRange(), beforeBody);

    // Handle cleanup region if it exists.
    // The cleanup region contains destructor calls that must run when exiting
    // the scope. We inline it after the scope body.
    mlir::Block *cleanupEntryBlock = nullptr;
    if (!scopeOp.getCleanupRegion().empty()) {
      cleanupEntryBlock = &scopeOp.getCleanupRegion().front();
      mlir::Block *cleanupLastBlock = &scopeOp.getCleanupRegion().back();

      // If we need to pass block arguments through cleanup, add them to the
      // cleanup entry block so they can be forwarded to the continue block.
      if (useBlockArgs) {
        cleanupEntryBlock->addArguments(scopeOp.getResultTypes(),
                                        llvm::SmallVector<mlir::Location>(
                                            scopeOp.getNumResults(), loc));
      }

      // Inline cleanup region before continue block
      rewriter.inlineRegionBefore(scopeOp.getCleanupRegion(), continueBlock);

      // Replace the cleanup yield with a branch to continue block
      if (auto cleanupYield =
              dyn_cast<cir::YieldOp>(cleanupLastBlock->getTerminator())) {
        rewriter.setInsertionPointToEnd(cleanupLastBlock);
        if (useBlockArgs) {
          // Pass the cleanup entry block arguments to continue block
          rewriter.replaceOpWithNewOp<cir::BrOp>(
              cleanupYield, cleanupEntryBlock->getArguments(), continueBlock);
        } else {
          rewriter.replaceOpWithNewOp<cir::BrOp>(cleanupYield,
                                                 mlir::ValueRange(),
                                                 continueBlock);
        }
      }
    }

    // Determine the target block for the scope body exit.
    // If there's a cleanup region, branch to it; otherwise, branch to continue.
    mlir::Block *bodyExitTarget =
        cleanupEntryBlock ? cleanupEntryBlock : continueBlock;

    // Replace the scopeop return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    // Note: afterBody pointer may be invalid after inlineRegionBefore,
    // need to get the terminator from the now-inlined block
    mlir::Block *inlinedLastBlock =
        cleanupEntryBlock ? cleanupEntryBlock->getPrevNode()
                          : continueBlock->getPrevNode();
    rewriter.setInsertionPointToEnd(inlinedLastBlock);
    if (auto inlinedYieldOp =
            dyn_cast<cir::YieldOp>(inlinedLastBlock->getTerminator())) {
      if (useBlockArgs) {
        // Pass the yield args to either cleanup or continue block
        rewriter.replaceOpWithNewOp<cir::BrOp>(inlinedYieldOp,
                                               inlinedYieldOp.getArgs(),
                                               bodyExitTarget);
      } else {
        // No block arguments - just branch
        rewriter.replaceOpWithNewOp<cir::BrOp>(inlinedYieldOp,
                                               mlir::ValueRange(),
                                               bodyExitTarget);
      }
    }

    // TODO(cir): stackrestore?

    // Replace the op with values return from the body region.
    if (useBlockArgs) {
      rewriter.replaceOp(scopeOp, continueBlock->getArguments());
    } else if (scopeOp.getNumResults() > 0) {
      // For materialized temporaries: the scope result is the address of a
      // local temporary. The pattern is:
      //   %alloca = cir.alloca ...
      //   %result = cir.scope {
      //     ... store/call to %alloca ...
      //   } : !cir.ptr<...>
      //   ... use %result (which is %alloca) ...
      //
      // We need to find what alloca the scope is returning. Look for a value
      // that's used as the first argument to a constructor-like call or as
      // the destination of a store.
      mlir::Value replacement;

      // Walk through the inlined block looking for the "temporary" address
      for (auto &op : *inlinedLastBlock) {
        if (auto storeOp = dyn_cast<cir::StoreOp>(&op)) {
          // The store destination might be the temporary
          auto addr = storeOp.getAddr();
          if (addr.getType() == scopeOp.getResultTypes()[0]) {
            replacement = addr;
            // Don't break - a later store with matching type is better
          }
        } else if (auto callOp = dyn_cast<cir::CallOp>(&op)) {
          // Constructor calls typically have the object pointer as first arg
          // and return void
          if (callOp.getNumOperands() > 0 &&
              callOp.getOperand(0).getType() == scopeOp.getResultTypes()[0]) {
            replacement = callOp.getOperand(0);
            // Don't break - continue looking for a better match
          }
        }
      }

      if (replacement) {
        rewriter.replaceOp(scopeOp, replacement);
      } else if (scopeOp.use_empty()) {
        // Scope result is unused - just erase the scope
        rewriter.eraseOp(scopeOp);
      } else {
        // Couldn't find the replacement value. This shouldn't happen.
        scopeOp.emitError("cannot flatten scope with result but no yield args");
        return mlir::failure();
      }
    } else {
      rewriter.eraseOp(scopeOp);
    }

    return mlir::success();
  }
};

class CIRTryOpFlattening : public mlir::OpRewritePattern<cir::TryOp> {
public:
  using OpRewritePattern<cir::TryOp>::OpRewritePattern;

  mlir::Block *buildTypeCase(mlir::PatternRewriter &rewriter, mlir::Region &r,
                             mlir::Block *afterTry,
                             mlir::Type exceptionPtrTy) const {
    YieldOp yieldOp;
    CatchParamOp paramOp;
    r.walk([&](YieldOp op) {
      assert(!yieldOp && "expect to only find one");
      yieldOp = op;
    });
    r.walk([&](CatchParamOp op) {
      assert(!paramOp && "expect to only find one");
      paramOp = op;
    });
    rewriter.inlineRegionBefore(r, afterTry);

    // Rewrite `cir.catch_param` to be scope aware and instead generate:
    // ```
    //   cir.catch_param begin %exception_ptr
    //   ...
    //   cir.catch_param end
    //   cir.br ...
    mlir::Value catchResult = paramOp.getParam();
    assert(catchResult && "expected to be available");
    rewriter.setInsertionPointAfterValue(catchResult);
    auto catchType = catchResult.getType();
    mlir::Block *entryBlock = paramOp->getBlock();
    mlir::Location catchLoc = paramOp.getLoc();
    // Catch handler only gets the exception pointer (selection not needed).
    mlir::Value exceptionPtr =
        entryBlock->addArgument(exceptionPtrTy, paramOp.getLoc());

    rewriter.replaceOpWithNewOp<cir::CatchParamOp>(
        paramOp, catchType, exceptionPtr,
        cir::CatchParamKindAttr::get(rewriter.getContext(),
                                     cir::CatchParamKind::Begin));

    rewriter.setInsertionPoint(yieldOp);
    rewriter.create<cir::CatchParamOp>(
        catchLoc, mlir::Type{}, nullptr,
        cir::CatchParamKindAttr::get(rewriter.getContext(),
                                     cir::CatchParamKind::End));

    rewriter.setInsertionPointToEnd(yieldOp->getBlock());
    rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, afterTry);
    return entryBlock;
  }

  void buildUnwindCase(mlir::PatternRewriter &rewriter, mlir::Region &r,
                       mlir::Block *unwindBlock) const {
    assert(&r.front() == &r.back() && "only one block expected");
    rewriter.mergeBlocks(&r.back(), unwindBlock);
    auto resume = dyn_cast<cir::ResumeOp>(unwindBlock->getTerminator());
    assert(resume && "expected 'cir.resume'");
    rewriter.setInsertionPointToEnd(unwindBlock);
    rewriter.replaceOpWithNewOp<cir::ResumeOp>(
        resume, unwindBlock->getArgument(0), unwindBlock->getArgument(1));
  }

  void buildAllCase(mlir::PatternRewriter &rewriter, mlir::Region &r,
                    mlir::Block *afterTry, mlir::Block *catchAllBlock,
                    mlir::Value exceptionPtr) const {
    YieldOp yieldOp;
    CatchParamOp paramOp;
    r.walk([&](YieldOp op) {
      assert(!yieldOp && "expect to only find one");
      yieldOp = op;
    });
    r.walk([&](CatchParamOp op) {
      assert(!paramOp && "expect to only find one");
      paramOp = op;
    });
    mlir::Block *catchAllStartBB = &r.front();
    rewriter.inlineRegionBefore(r, afterTry);
    rewriter.mergeBlocks(catchAllStartBB, catchAllBlock);

    // Rewrite `cir.catch_param` to be scope aware and instead generate:
    // ```
    //   cir.catch_param begin %exception_ptr
    //   ...
    //   cir.catch_param end
    //   cir.br ...
    mlir::Value catchResult = paramOp.getParam();
    assert(catchResult && "expected to be available");
    rewriter.setInsertionPointAfterValue(catchResult);
    auto catchType = catchResult.getType();
    mlir::Location catchLoc = paramOp.getLoc();
    rewriter.replaceOpWithNewOp<cir::CatchParamOp>(
        paramOp, catchType, exceptionPtr,
        cir::CatchParamKindAttr::get(rewriter.getContext(),
                                     cir::CatchParamKind::Begin));

    rewriter.setInsertionPoint(yieldOp);
    rewriter.create<cir::CatchParamOp>(
        catchLoc, mlir::Type{}, nullptr,
        cir::CatchParamKindAttr::get(rewriter.getContext(),
                                     cir::CatchParamKind::End));

    rewriter.setInsertionPointToEnd(yieldOp->getBlock());
    rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, afterTry);
  }

  mlir::ArrayAttr collectTypeSymbols(cir::TryOp tryOp) const {
    mlir::ArrayAttr caseAttrList = tryOp.getCatchTypesAttr();
    llvm::SmallVector<mlir::Attribute, 4> symbolList;

    for (mlir::Attribute caseAttr : caseAttrList) {
      auto typeIdGlobal = dyn_cast<cir::GlobalViewAttr>(caseAttr);
      if (!typeIdGlobal)
        continue;
      symbolList.push_back(typeIdGlobal.getSymbol());
    }

    // Return an empty attribute instead of an empty list...
    if (symbolList.empty())
      return {};
    return mlir::ArrayAttr::get(caseAttrList.getContext(), symbolList);
  }

  void buildLandingPad(cir::TryOp tryOp, mlir::PatternRewriter &rewriter,
                       mlir::Block *beforeCatch, mlir::Block *landingPadBlock,
                       mlir::Block *catchDispatcher,
                       SmallVectorImpl<cir::CallOp> &callsToRewrite,
                       unsigned callIdx, bool tryOnlyHasCatchAll,
                       mlir::Type exceptionPtrType,
                       mlir::Type typeIdType) const {
    rewriter.setInsertionPointToEnd(landingPadBlock);
    mlir::ArrayAttr symlist = collectTypeSymbols(tryOp);
    auto inflightEh = rewriter.create<cir::EhInflightOp>(
        tryOp.getLoc(), exceptionPtrType, typeIdType,
        tryOp.getCleanup() ? mlir::UnitAttr::get(tryOp.getContext()) : nullptr,
        symlist);
    auto selector = inflightEh.getTypeId();
    auto exceptionPtr = inflightEh.getExceptionPtr();

    // Time to emit cleanup's.
    cir::CallOp callOp = callsToRewrite[callIdx];
    if (!callOp.getCleanup().empty()) {
      mlir::Block *cleanupBlock = &callOp.getCleanup().getBlocks().back();
      auto cleanupYield = cast<cir::YieldOp>(cleanupBlock->getTerminator());
      rewriter.eraseOp(cleanupYield);
      rewriter.mergeBlocks(cleanupBlock, landingPadBlock);
      rewriter.setInsertionPointToEnd(landingPadBlock);
    }

    // Branch out to the catch clauses dispatcher.
    assert(catchDispatcher->getNumArguments() >= 1 &&
           "expected at least one argument in place");
    llvm::SmallVector<mlir::Value> dispatcherInitOps = {exceptionPtr};
    if (!tryOnlyHasCatchAll) {
      assert(catchDispatcher->getNumArguments() == 2 &&
             "expected two arguments in place");
      dispatcherInitOps.push_back(selector);
    }
    rewriter.create<cir::BrOp>(tryOp.getLoc(), catchDispatcher,
                               dispatcherInitOps);
    return;
  }

  mlir::Block *buildLandingPads(cir::TryOp tryOp,
                                mlir::PatternRewriter &rewriter,
                                mlir::Block *beforeCatch, mlir::Block *afterTry,
                                SmallVectorImpl<cir::CallOp> &callsToRewrite,
                                SmallVectorImpl<mlir::Block *> &landingPads,
                                bool tryOnlyHasCatchAll) const {
    unsigned numCalls = callsToRewrite.size();
    // Create the first landing pad block and a placeholder for the initial
    // catch dispatcher (which will be the common destination for every new
    // landing pad we create).
    auto *landingPadBlock =
        rewriter.splitBlock(beforeCatch, rewriter.getInsertionPoint());

    // For the dispatcher, already add the block arguments and prepare the
    // proper types the landing pad should use to jump to.
    mlir::Block *dispatcher = rewriter.createBlock(afterTry);
    auto exceptionPtrType =
        cir::PointerType::get(cir::VoidType::get(rewriter.getContext()));
    auto typeIdType = cir::IntType::get(getContext(), 32, false);
    dispatcher->addArgument(exceptionPtrType, tryOp.getLoc());
    if (!tryOnlyHasCatchAll)
      dispatcher->addArgument(typeIdType, tryOp.getLoc());

    for (unsigned callIdx = 0; callIdx != numCalls; ++callIdx) {
      buildLandingPad(tryOp, rewriter, beforeCatch, landingPadBlock, dispatcher,
                      callsToRewrite, callIdx, tryOnlyHasCatchAll,
                      exceptionPtrType, typeIdType);
      landingPads.push_back(landingPadBlock);
      if (callIdx < numCalls - 1)
        landingPadBlock = rewriter.createBlock(dispatcher);
    }

    return dispatcher;
  }

  mlir::Block *buildCatch(cir::TryOp tryOp, mlir::PatternRewriter &rewriter,
                          mlir::Block *afterTry, mlir::Block *dispatcher,
                          SmallVectorImpl<cir::CallOp> &callsToRewrite,
                          mlir::Attribute catchAttr,
                          mlir::Attribute nextCatchAttr,
                          mlir::Region &catchRegion) const {
    mlir::Location loc = tryOp.getLoc();
    mlir::Block *nextDispatcher = nullptr;
    if (auto typeIdGlobal = dyn_cast<cir::GlobalViewAttr>(catchAttr)) {
      auto *previousDispatcher = dispatcher;
      auto typeId =
          rewriter.create<cir::EhTypeIdOp>(loc, typeIdGlobal.getSymbol());
      auto ehPtr = previousDispatcher->getArgument(0);
      auto ehSel = previousDispatcher->getArgument(1);

      auto match = rewriter.create<cir::CmpOp>(
          loc, cir::BoolType::get(rewriter.getContext()), cir::CmpOpKind::eq,
          ehSel, typeId);

      mlir::Block *typeCatchBlock =
          buildTypeCase(rewriter, catchRegion, afterTry, ehPtr.getType());
      nextDispatcher = rewriter.createBlock(afterTry);
      rewriter.setInsertionPointToEnd(previousDispatcher);

      // Next dispatcher gets by default both exception ptr and selector info,
      // but on a catch all we don't need selector info.
      nextDispatcher->addArgument(ehPtr.getType(), loc);
      llvm::SmallVector<mlir::Value> nextDispatchOps = {ehPtr};
      if (!isa<cir::CatchAllAttr>(nextCatchAttr)) {
        nextDispatcher->addArgument(ehSel.getType(), loc);
        nextDispatchOps.push_back(ehSel);
      }

      rewriter.create<cir::BrCondOp>(loc, match, typeCatchBlock, nextDispatcher,
                                     mlir::ValueRange{ehPtr}, nextDispatchOps);
      rewriter.setInsertionPointToEnd(nextDispatcher);
    } else if (auto catchAll = dyn_cast<cir::CatchAllAttr>(catchAttr)) {
      // In case the catch(...) is all we got, `dispatcher` shall be
      // non-empty.
      assert(dispatcher->getArguments().size() == 1 &&
             "expected one block argument");
      auto ehPtr = dispatcher->getArgument(0);
      buildAllCase(rewriter, catchRegion, afterTry, dispatcher, ehPtr);
      // Do not update `nextDispatcher`, no more business in try/catch
    } else if (auto catchUnwind = dyn_cast<cir::CatchUnwindAttr>(catchAttr)) {
      assert(dispatcher->getArguments().size() == 2 &&
             "expected two block argument");
      buildUnwindCase(rewriter, catchRegion, dispatcher);
      // Do not update `nextDispatcher`, no more business in try/catch
    }
    return nextDispatcher;
  }

  void buildCatchers(cir::TryOp tryOp, mlir::PatternRewriter &rewriter,
                     mlir::Block *afterBody, mlir::Block *afterTry,
                     SmallVectorImpl<cir::CallOp> &callsToRewrite,
                     SmallVectorImpl<mlir::Block *> &landingPads) const {
    // Replace the tryOp return with a branch that jumps out of the body.
    rewriter.setInsertionPointToEnd(afterBody);

    mlir::Block *beforeCatch = rewriter.getInsertionBlock();
    rewriter.setInsertionPointToEnd(beforeCatch);

    // Check if the terminator is a YieldOp because there could be another
    // terminator, e.g. unreachable
    if (auto tryBodyYield = dyn_cast<cir::YieldOp>(afterBody->getTerminator()))
      rewriter.replaceOpWithNewOp<cir::BrOp>(tryBodyYield, afterTry);

    mlir::ArrayAttr catches = tryOp.getCatchTypesAttr();
    if (!catches || catches.empty()) {
      // No catch clauses - can't build landing pads, so don't rewrite calls.
      // The calls will remain as cir.call (not exception throwing) since
      // exception handling infrastructure won't be generated for this try.
      callsToRewrite.clear();
      return;
    }

    // If there are no calls that can throw exceptions, skip building
    // exception handling infrastructure. The try body is already inlined,
    // and without any exceptional calls, the catch blocks are dead code.
    if (callsToRewrite.empty())
      return;

    // Start the landing pad by getting the inflight exception information.
    mlir::Block *nextDispatcher =
        buildLandingPads(tryOp, rewriter, beforeCatch, afterTry, callsToRewrite,
                         landingPads, tryOp.isCatchAllOnly());

    // Fill in dispatcher to all catch clauses.
    rewriter.setInsertionPointToEnd(nextDispatcher);
    llvm::MutableArrayRef<mlir::Region> catchRegions = tryOp.getCatchRegions();
    unsigned catchIdx = 0;

    // Build control-flow for all catch clauses.
    mlir::ArrayAttr catchAttrList = tryOp.getCatchTypesAttr();
    for (mlir::Attribute catchAttr : catchAttrList) {
      mlir::Attribute nextCatchAttr;
      if (catchIdx + 1 < catchAttrList.size())
        nextCatchAttr = catchAttrList[catchIdx + 1];
      nextDispatcher =
          buildCatch(tryOp, rewriter, afterTry, nextDispatcher, callsToRewrite,
                     catchAttr, nextCatchAttr, catchRegions[catchIdx]);
      catchIdx++;
    }

    assert(!nextDispatcher && "last dispatch expected to be nullptr");
  }

  mlir::Block *buildTryBody(cir::TryOp tryOp,
                            mlir::PatternRewriter &rewriter) const {
    auto loc = tryOp.getLoc();
    // Split the current block before the TryOp to create the inlining
    // point.
    auto *beforeTryScopeBlock = rewriter.getInsertionBlock();
    mlir::Block *afterTry =
        rewriter.splitBlock(beforeTryScopeBlock, rewriter.getInsertionPoint());

    // Inline body region.
    auto *beforeBody = &tryOp.getTryRegion().front();
    rewriter.inlineRegionBefore(tryOp.getTryRegion(), afterTry);

    // Branch into the body of the region.
    rewriter.setInsertionPointToEnd(beforeTryScopeBlock);
    rewriter.create<cir::BrOp>(loc, mlir::ValueRange(), beforeBody);
    return afterTry;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::TryOp tryOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto *afterBody = &tryOp.getTryRegion().back();

    // Empty scope: just remove it.
    if (tryOp.getTryRegion().empty()) {
      rewriter.eraseOp(tryOp);
      return mlir::success();
    }

    // Grab the collection of `cir.call exception`s to rewrite to
    // `cir.try_call`.
    llvm::SmallVector<cir::CallOp, 4> callsToRewrite;
    tryOp.getTryRegion().walk([&](CallOp op) {
      // Only grab calls within immediate closest TryOp scope.
      if (op->getParentOfType<cir::TryOp>() != tryOp)
        return;
      if (!op.getException())
        return;
      callsToRewrite.push_back(op);
    });

    // Build try body.
    mlir::Block *afterTry = buildTryBody(tryOp, rewriter);

    // Build catchers.
    llvm::SmallVector<mlir::Block *, 4> landingPads;
    buildCatchers(tryOp, rewriter, afterBody, afterTry, callsToRewrite,
                  landingPads);
    rewriter.eraseOp(tryOp);
    assert((landingPads.size() == callsToRewrite.size()) &&
           "expected matching number of entries");

    // Rewrite calls.
    unsigned callIdx = 0;
    for (CallOp callOp : callsToRewrite) {
      mlir::Block *callBlock = callOp->getBlock();
      mlir::Block *cont =
          rewriter.splitBlock(callBlock, mlir::Block::iterator(callOp));
      cir::ExtraFuncAttributesAttr extraAttrs = callOp.getExtraAttrs();
      std::optional<cir::ASTCallExprInterface> ast = callOp.getAst();

      mlir::FlatSymbolRefAttr symbol;
      if (!callOp.isIndirect())
        symbol = callOp.getCalleeAttr();
      rewriter.setInsertionPointToEnd(callBlock);
      mlir::Type resTy = nullptr;
      if (callOp.getNumResults() > 0)
        resTy = callOp.getResult().getType();
      auto tryCall = rewriter.replaceOpWithNewOp<cir::TryCallOp>(
          callOp, symbol, resTy, cont, landingPads[callIdx],
          callOp.getOperands());
      tryCall.setExtraAttrsAttr(extraAttrs);
      if (ast)
        tryCall.setAstAttr(*ast);
      callIdx++;
    }

    // Quick block cleanup: no indirection to the post try block.
    auto brOp = dyn_cast<cir::BrOp>(afterTry->getTerminator());
    if (brOp && brOp.getDest()->hasNoPredecessors()) {
      mlir::Block *srcBlock = brOp.getDest();
      rewriter.eraseOp(brOp);
      rewriter.mergeBlocks(srcBlock, afterTry);
    }
    return mlir::success();
  }
};

class CIRLoopOpInterfaceFlattening
    : public mlir::OpInterfaceRewritePattern<cir::LoopOpInterface> {
public:
  using mlir::OpInterfaceRewritePattern<
      cir::LoopOpInterface>::OpInterfaceRewritePattern;

  inline void lowerConditionOp(cir::ConditionOp op, mlir::Block *body,
                               mlir::Block *exit,
                               mlir::PatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cir::BrCondOp>(op, op.getCondition(), body,
                                               exit);
  }

  mlir::LogicalResult
  matchAndRewrite(cir::LoopOpInterface op,
                  mlir::PatternRewriter &rewriter) const final {
    // Setup CFG blocks.
    auto *entry = rewriter.getInsertionBlock();
    auto *exit = rewriter.splitBlock(entry, rewriter.getInsertionPoint());
    auto *cond = &op.getCond().front();
    auto *body = &op.getBody().front();
    auto *step = (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);

    // Setup loop entry branch.
    rewriter.setInsertionPointToEnd(entry);
    rewriter.create<cir::BrOp>(op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit.
    auto conditionOp = cast<cir::ConditionOp>(cond->getTerminator());
    lowerConditionOp(conditionOp, body, exit, rewriter);

    // TODO(cir): Remove the walks below. It visits operations unnecessarily,
    // however, to solve this we would likely need a custom DialecConversion
    // driver to customize the order that operations are visited.

    // Lower continue statements.
    mlir::Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](mlir::Operation *op) {
      if (!isa<cir::ContinueOp>(op))
        return mlir::WalkResult::advance();

      lowerTerminator(op, dest, rewriter);
      return mlir::WalkResult::skip();
    });

    // Lower break statements.
    walkRegionSkipping<cir::LoopOpInterface, cir::SwitchOp>(
        op.getBody(), [&](mlir::Operation *op) {
          if (!isa<cir::BreakOp>(op))
            return mlir::WalkResult::advance();

          lowerTerminator(op, exit, rewriter);
          return mlir::WalkResult::skip();
        });

    // Lower optional body region yield.
    for (auto &blk : op.getBody().getBlocks()) {
      auto bodyYield = dyn_cast<cir::YieldOp>(blk.getTerminator());
      if (bodyYield)
        lowerTerminator(bodyYield, (step ? step : cond), rewriter);
    }

    // Lower mandatory step region yield.
    if (step)
      lowerTerminator(cast<cir::YieldOp>(step->getTerminator()), cond,
                      rewriter);

    // Move region contents out of the loop op.
    rewriter.inlineRegionBefore(op.getCond(), exit);
    rewriter.inlineRegionBefore(op.getBody(), exit);
    if (step)
      rewriter.inlineRegionBefore(*op.maybeGetStep(), exit);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRSwitchOpFlattening : public mlir::OpRewritePattern<cir::SwitchOp> {
public:
  using OpRewritePattern<cir::SwitchOp>::OpRewritePattern;

  inline void rewriteYieldOp(mlir::PatternRewriter &rewriter,
                             cir::YieldOp yieldOp,
                             mlir::Block *destination) const {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<cir::BrOp>(yieldOp, yieldOp.getOperands(),
                                           destination);
  }

  // Return the new defaultDestination block.
  Block *condBrToRangeDestination(cir::SwitchOp op,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Block *rangeDestination,
                                  mlir::Block *defaultDestination,
                                  APInt lowerBound, APInt upperBound) const {
    assert(lowerBound.sle(upperBound) && "Invalid range");
    auto resBlock = rewriter.createBlock(defaultDestination);
    auto sIntType = cir::IntType::get(op.getContext(), 32, true);
    auto uIntType = cir::IntType::get(op.getContext(), 32, false);

    auto rangeLength = rewriter.create<cir::ConstantOp>(
        op.getLoc(), cir::IntAttr::get(sIntType, upperBound - lowerBound));

    auto lowerBoundValue = rewriter.create<cir::ConstantOp>(
        op.getLoc(), cir::IntAttr::get(sIntType, lowerBound));
    auto diffValue =
        rewriter.create<cir::BinOp>(op.getLoc(), sIntType, cir::BinOpKind::Sub,
                                    op.getCondition(), lowerBoundValue);

    // Use unsigned comparison to check if the condition is in the range.
    auto uDiffValue = rewriter.create<cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, diffValue);
    auto uRangeLength = rewriter.create<cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, rangeLength);

    auto cmpResult = rewriter.create<cir::CmpOp>(
        op.getLoc(), cir::BoolType::get(op.getContext()), cir::CmpOpKind::le,
        uDiffValue, uRangeLength);
    rewriter.create<cir::BrCondOp>(op.getLoc(), cmpResult, rangeDestination,
                                   defaultDestination);
    return resBlock;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::SwitchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<CaseOp> cases;
    op.collectCases(cases);

    // Empty switch statement: just erase it.
    if (cases.empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Create exit block from the next node of cir.switch op.
    auto *exitBlock = rewriter.splitBlock(rewriter.getBlock(),
                                          op->getNextNode()->getIterator());

    // We lower cir.switch op in the following process:
    // 1. Inline the region from the switch op after switch op.
    // 2. Traverse each cir.case op:
    //    a. Record the entry block, block arguments and condition for every
    //    case. b. Inline the case region after the case op.
    // 3. Replace the empty cir.switch.op with the new cir.switchflat op by the
    //    recorded block and conditions.

    // inline everything from switch body between the switch op and the exit
    // block.
    {
      cir::YieldOp switchYield = nullptr;
      // Clear switch operation.
      for (auto &block : llvm::make_early_inc_range(op.getBody().getBlocks()))
        if (auto yieldOp = dyn_cast<cir::YieldOp>(block.getTerminator()))
          switchYield = yieldOp;

      assert(!op.getBody().empty());
      mlir::Block *originalBlock = op->getBlock();
      mlir::Block *swopBlock =
          rewriter.splitBlock(originalBlock, op->getIterator());
      rewriter.inlineRegionBefore(op.getBody(), exitBlock);

      if (switchYield)
        rewriteYieldOp(rewriter, switchYield, exitBlock);

      rewriter.setInsertionPointToEnd(originalBlock);
      rewriter.create<cir::BrOp>(op.getLoc(), swopBlock);
    }

    // Allocate required data structures (disconsider default case in
    // vectors).
    llvm::SmallVector<mlir::APInt, 8> caseValues;
    llvm::SmallVector<mlir::Block *, 8> caseDestinations;
    llvm::SmallVector<mlir::ValueRange, 8> caseOperands;

    llvm::SmallVector<std::pair<APInt, APInt>> rangeValues;
    llvm::SmallVector<mlir::Block *> rangeDestinations;
    llvm::SmallVector<mlir::ValueRange> rangeOperands;

    // Initialize default case as optional.
    mlir::Block *defaultDestination = exitBlock;
    mlir::ValueRange defaultOperands = exitBlock->getArguments();

    // Digest the case statements values and bodies.
    for (auto caseOp : cases) {
      mlir::Region &region = caseOp.getCaseRegion();

      // Found default case: save destination and operands.
      switch (caseOp.getKind()) {
      case cir::CaseOpKind::Default:
        defaultDestination = &region.front();
        defaultOperands = defaultDestination->getArguments();
        break;
      case cir::CaseOpKind::Range:
        assert(caseOp.getValue().size() == 2 &&
               "Case range should have 2 case value");
        rangeValues.push_back(
            {cast<cir::IntAttr>(caseOp.getValue()[0]).getValue(),
             cast<cir::IntAttr>(caseOp.getValue()[1]).getValue()});
        rangeDestinations.push_back(&region.front());
        rangeOperands.push_back(rangeDestinations.back()->getArguments());
        break;
      case cir::CaseOpKind::Anyof:
      case cir::CaseOpKind::Equal:
        // AnyOf cases kind can have multiple values, hence the loop below.
        for (auto &value : caseOp.getValue()) {
          caseValues.push_back(cast<cir::IntAttr>(value).getValue());
          caseDestinations.push_back(&region.front());
          caseOperands.push_back(caseDestinations.back()->getArguments());
        }
        break;
      }

      // Handle break statements.
      walkRegionSkipping<cir::LoopOpInterface, cir::SwitchOp>(
          region, [&](mlir::Operation *op) {
            if (!isa<cir::BreakOp>(op))
              return mlir::WalkResult::advance();

            lowerTerminator(op, exitBlock, rewriter);
            return mlir::WalkResult::skip();
          });

      // Track fallthrough in cases.
      for (auto &blk : region.getBlocks()) {
        if (blk.getNumSuccessors())
          continue;

        if (auto yieldOp = dyn_cast<cir::YieldOp>(blk.getTerminator())) {
          mlir::Operation *nextOp = caseOp->getNextNode();
          assert(nextOp && "caseOp is not expected to be the last op");
          mlir::Block *oldBlock = nextOp->getBlock();
          mlir::Block *newBlock =
              rewriter.splitBlock(oldBlock, nextOp->getIterator());
          rewriter.setInsertionPointToEnd(oldBlock);
          rewriter.create<cir::BrOp>(nextOp->getLoc(), mlir::ValueRange(),
                                     newBlock);
          rewriteYieldOp(rewriter, yieldOp, newBlock);
        }
      }

      mlir::Block *oldBlock = caseOp->getBlock();
      mlir::Block *newBlock =
          rewriter.splitBlock(oldBlock, caseOp->getIterator());

      mlir::Block &entryBlock = caseOp.getCaseRegion().front();
      rewriter.inlineRegionBefore(caseOp.getCaseRegion(), newBlock);

      // Create a branch to the entry of the inlined region.
      rewriter.setInsertionPointToEnd(oldBlock);
      rewriter.create<cir::BrOp>(caseOp.getLoc(), &entryBlock);
    }

    // Remove all cases since we've inlined the regions.
    for (auto caseOp : cases) {
      mlir::Block *caseBlock = caseOp->getBlock();
      // Erase the block with no predecessors here to make the generated code
      // simpler a little bit.
      if (caseBlock->hasNoPredecessors())
        rewriter.eraseBlock(caseBlock);
      else
        rewriter.eraseOp(caseOp);
    }

    for (size_t index = 0; index < rangeValues.size(); ++index) {
      auto lowerBound = rangeValues[index].first;
      auto upperBound = rangeValues[index].second;

      // The case range is unreachable, skip it.
      if (lowerBound.sgt(upperBound))
        continue;

      // If range is small, add multiple switch instruction cases.
      // This magical number is from the original CGStmt code.
      constexpr int kSmallRangeThreshold = 64;
      if ((upperBound - lowerBound)
              .ult(llvm::APInt(32, kSmallRangeThreshold))) {
        for (auto iValue = lowerBound; iValue.sle(upperBound); (void)iValue++) {
          caseValues.push_back(iValue);
          caseOperands.push_back(rangeOperands[index]);
          caseDestinations.push_back(rangeDestinations[index]);
        }
        continue;
      }

      defaultDestination =
          condBrToRangeDestination(op, rewriter, rangeDestinations[index],
                                   defaultDestination, lowerBound, upperBound);
      defaultOperands = rangeOperands[index];
    }

    // Set switch op to branch to the newly created blocks.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cir::SwitchFlatOp>(
        op, op.getCondition(), defaultDestination, defaultOperands, caseValues,
        caseDestinations, caseOperands);

    return mlir::success();
  }
};
class CIRTernaryOpFlattening : public mlir::OpRewritePattern<cir::TernaryOp> {
public:
  using OpRewritePattern<cir::TernaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TernaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);

    auto &trueRegion = op.getTrueRegion();
    auto *trueBlock = &trueRegion.front();
    mlir::Operation *trueTerminator = trueRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&trueRegion.back());
    auto trueYieldOp = dyn_cast<cir::YieldOp>(trueTerminator);

    rewriter.replaceOpWithNewOp<cir::BrOp>(trueYieldOp, trueYieldOp.getArgs(),
                                           remainingOpsBlock);
    rewriter.inlineRegionBefore(trueRegion, remainingOpsBlock);

    auto &falseRegion = op.getFalseRegion();
    auto *falseBlock = &falseRegion.front();

    mlir::Operation *falseTerminator = falseRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&falseRegion.back());
    auto falseYieldOp = dyn_cast<cir::YieldOp>(falseTerminator);
    rewriter.replaceOpWithNewOp<cir::BrOp>(falseYieldOp, falseYieldOp.getArgs(),
                                           remainingOpsBlock);
    rewriter.inlineRegionBefore(falseRegion, remainingOpsBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<cir::BrCondOp>(loc, op.getCond(), trueBlock, falseBlock);

    if (auto rt = op.getResultTypes(); rt.size()) {
      auto args = remainingOpsBlock->addArguments(rt, op.getLoc());
      SmallVector<mlir::Value, 2> values;
      llvm::copy(args, std::back_inserter(values));
      rewriter.replaceOpUsesWithinBlock(op, values, remainingOpsBlock);
    }
    rewriter.eraseOp(op);

    // Ok, we're done!
    return mlir::success();
  }
};

void populateFlattenCFGPatterns(RewritePatternSet &patterns) {
  patterns
      .add<CIRIfFlattening, CIRLoopOpInterfaceFlattening, CIRScopeOpFlattening,
           CIRSwitchOpFlattening, CIRTernaryOpFlattening, CIRTryOpFlattening>(
          patterns.getContext());
}

void FlattenCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFlattenCFGPatterns(patterns);

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, ScopeOp, SwitchOp, LoopOpInterface, TernaryOp, TryOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsGreedily(ops, std::move(patterns)).failed())
    signalPassFailure();

  // Remove stray yield operations that appear after terminators.
  // This can happen when a block is split and the original yield is left
  // orphaned after a branch instruction.
  getOperation()->walk([](cir::FuncOp funcOp) {
    for (auto &block : funcOp.getBody()) {
      // Find the first terminator in the block
      mlir::Operation *firstTerminator = nullptr;
      for (auto &op : block) {
        if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
          firstTerminator = &op;
          break;
        }
      }

      // If we found a terminator and it's not the last op, remove everything
      // after it (these are stray ops that shouldn't be there)
      if (firstTerminator && firstTerminator != &block.back()) {
        llvm::SmallVector<mlir::Operation *, 4> toRemove;
        bool afterTerminator = false;
        for (auto &op : block) {
          if (afterTerminator)
            toRemove.push_back(&op);
          if (&op == firstTerminator)
            afterTerminator = true;
        }
        for (auto *op : toRemove)
          op->erase();
      }
    }
  });

  // Fix orphaned allocas at module scope.
  // These can occur due to bugs in CIR codegen where allocas leak outside
  // of function bodies. Such allocas are invalid (they're not inside any
  // function and cannot be used) and must be fixed to prevent crashes
  // during lowering to LLVM.
  //
  // For orphaned allocas with uses, we clone them into each function that
  // uses them. For unused ones, we simply remove them.
  if (auto modOp = dyn_cast<mlir::ModuleOp>(getOperation())) {
    llvm::SmallVector<cir::AllocaOp, 4> orphanedAllocas;
    for (auto &op : modOp.getBody()->getOperations()) {
      if (auto alloca = dyn_cast<cir::AllocaOp>(&op)) {
        // An alloca at module scope is orphaned - it should be inside a
        // function. Collect it for fixing.
        orphanedAllocas.push_back(alloca);
      }
    }

    for (auto alloca : orphanedAllocas) {
      if (alloca.use_empty()) {
        // Unused orphaned alloca - simply remove it.
        alloca.erase();
        continue;
      }

      // Group uses by the containing function.
      llvm::DenseMap<cir::FuncOp, llvm::SmallVector<mlir::OpOperand *, 4>>
          usesByFunc;
      for (auto &use : alloca->getUses()) {
        if (auto funcOp = use.getOwner()->getParentOfType<cir::FuncOp>()) {
          usesByFunc[funcOp].push_back(&use);
        }
      }

      // For each function that uses this alloca, clone it into the function's
      // entry block and update the uses.
      for (auto &[funcOp, uses] : usesByFunc) {
        if (funcOp.getRegion().empty())
          continue;

        mlir::Block &entryBlock = funcOp.getRegion().front();
        mlir::OpBuilder builder(&entryBlock, entryBlock.begin());

        // Clone the alloca into this function's entry block.
        auto clonedAlloca = builder.create<cir::AllocaOp>(
            alloca.getLoc(), alloca.getType(), alloca.getAllocaType(),
            alloca.getName(), alloca.getAlignmentAttr(),
            alloca.getDynAllocSize());
        clonedAlloca.setInit(alloca.getInit());
        clonedAlloca.setConstant(alloca.getConstant());

        // Update uses in this function to point to the cloned alloca.
        for (auto *use : uses) {
          use->set(clonedAlloca.getResult());
        }
      }

      // After cloning to all functions, the original should have no uses left.
      // Remove it.
      if (alloca.use_empty()) {
        alloca.erase();
      } else {
        // This shouldn't happen, but emit a warning if it does.
        alloca.emitWarning("orphaned alloca at module scope still has uses "
                           "after attempted fix");
      }
    }
  }

  // Remove dead blocks that may have been left behind after flattening.
  // These can occur when cleanup regions are inlined but their blocks
  // become orphaned (no predecessors).
  // NOTE: We must NOT remove blocks containing labels, as GotoSolver needs
  // them to resolve goto statements.
  getOperation()->walk([](cir::FuncOp funcOp) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &block : llvm::make_early_inc_range(funcOp.getBody())) {
        if (!block.isEntryBlock() && block.hasNoPredecessors()) {
          // Check if this block contains a label - if so, don't remove it
          // because GotoSolver needs to resolve gotos to this label.
          bool hasLabel = false;
          for (auto &op : block) {
            if (isa<cir::LabelOp>(op)) {
              hasLabel = true;
              break;
            }
          }
          if (hasLabel)
            continue;

          // Drop all references from this block before erasing.
          // This disconnects the terminator from its successors, which
          // prevents dangling predecessor references (INVALIDBLOCK).
          block.dropAllReferences();
          block.erase();
          changed = true;
        }
      }
    }
  });
}

} // namespace

namespace mlir {

std::unique_ptr<Pass> createFlattenCFGPass() {
  return std::make_unique<FlattenCFGPass>();
}

} // namespace mlir
