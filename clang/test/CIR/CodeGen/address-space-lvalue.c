// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

typedef unsigned __attribute__((address_space(1))) *remote_ptr;

// CIR-LABEL: cir.func dso_local @test_array_subscript
// CIR: %[[CAST:.*]] = cir.cast(address_space, %{{.*}} : !cir.ptr<!u32i, addrspace(target<1>)>), !cir.ptr<!u32i>
// CIR: %[[ELEM:.*]] = cir.ptr_stride %[[CAST]], %{{.*}} : (!cir.ptr<!u32i>, !u64i) -> !cir.ptr<!u32i>
// CIR: %{{.*}} = cir.load{{.*}} %[[ELEM]] : !cir.ptr<!u32i>, !u32i
// LLVM-LABEL: define dso_local i32 @test_array_subscript
// LLVM: %[[CAST:.*]] = addrspacecast ptr addrspace(1) %{{.*}} to ptr
// LLVM: %[[ELEM:.*]] = getelementptr i32, ptr %[[CAST]], i64 %{{.*}}
// LLVM: load i32, ptr %[[ELEM]], align 4
unsigned test_array_subscript(remote_ptr ptr, unsigned long index) {
  return ((unsigned *)ptr)[index];
}
