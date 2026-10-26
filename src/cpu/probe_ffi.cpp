#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static ffi::Error AddOne(ffi::Buffer<ffi::F32> x,
                         ffi::Result<ffi::Buffer<ffi::F32>> y) {
  const float* in = x.typed_data();
  float* out = y->typed_data();
  const int64_t n = x.element_count();
  for (int64_t i = 0; i < n; ++i) out[i] = in[i] + 1.0f;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(AddOneFFI, AddOne,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());
