#include <type_traits>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_benchmark.h"
#include "cuda/numeric"

template <typename data_type, int use_shared>
struct helper
{
  static __device__ cuda::warp_reduce<data_type> get_sync(data_type *shared_data)
  {
    return cuda::warp_reduce<data_type> (shared_data);
  }
};

template <typename data_type>
struct helper<data_type, false>
{
  static __device__ cuda::warp_reduce<data_type> get_sync(data_type *)
  {
    return cuda::warp_reduce<data_type> ();
  }
};


template <typename data_type>
__global__ void perform_reduce (const data_type *array, unsigned int n, data_type *result)
{
  extern __shared__ char cache_storage[];
  data_type *cache = reinterpret_cast<data_type*> (cache_storage);

  cuda::warp_reduce<data_type> reduce = helper<data_type, cuda::warp_reduce<data_type>::use_shared>::get_sync(cache);
  data_type value = reduce (array, array + n, data_type {});

  if (threadIdx.x < n)
    result[threadIdx.x] = value;
}

template <typename data_type>
__global__ void perform_value_reduce (const data_type *array, unsigned int n, data_type *result)
{
  extern __shared__ char cache_storage[];
  data_type *cache = reinterpret_cast<data_type*> (cache_storage);

  cuda::warp_reduce<data_type> reduce = helper<data_type, cuda::warp_reduce<data_type>::use_shared>::get_sync(cache);
  data_type value = reduce (array[threadIdx.x]);

  if (threadIdx.x < n)
    result[threadIdx.x] = value;
}

template <typename data_type>
__global__ void perform_block_value_reduce (const data_type *array, unsigned int n, data_type *result)
{
  extern __shared__ char cache_storage[];
  data_type *cache = reinterpret_cast<data_type*> (cache_storage);

  cuda::block_reduce<data_type> reduce (cache);
  data_type value = reduce (array[threadIdx.x]);

  if (threadIdx.x < n)
    result[threadIdx.x] = value;
}

template <typename data_type>
void expect_eq (
    const std::vector<data_type> input,
    const std::vector<data_type> &excected_output,
    unsigned int threads_per_block)
{
  data_type *device_input {};
  data_type *device_result {};

  cudaMalloc (&device_input, input.size () * sizeof (data_type));
  cudaMalloc (&device_result, input.size () * sizeof (data_type));

  cudaMemcpy (device_input, input.data (), input.size () * sizeof (data_type), cudaMemcpyHostToDevice);

  const int blocks_count = (input.size () + threads_per_block - 1) / threads_per_block;
  const size_t shared_mem_size = threads_per_block * blocks_count * sizeof (data_type);
  std::vector<data_type> output (input.size (), data_type {});

  if (threads_per_block == 32)
  {
    perform_reduce<<<blocks_count, threads_per_block, shared_mem_size>>>(device_input, input.size (), device_result);
    cudaMemcpy (output.data (), device_result, input.size () * sizeof (data_type), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < input.size (); i++)
      if (excected_output[i] != output[i])
        throw std::runtime_error ("Error: unexpected value at " + std::to_string (i));

    perform_value_reduce<<<blocks_count, threads_per_block, shared_mem_size>>>(device_input, input.size (), device_result);
    std::fill (output.begin (), output.end (), data_type {});

    cudaMemcpy (output.data (), device_result, input.size () * sizeof (data_type), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < input.size (); i++)
      if (excected_output[i] != output[i])
        throw std::runtime_error ("Error: unexpected value at " + std::to_string (i));
  }

  perform_block_value_reduce<<<blocks_count, threads_per_block, shared_mem_size>>>(device_input, input.size (), device_result);
  std::fill (output.begin (), output.end (), data_type {});

  cudaMemcpy (output.data (), device_result, input.size () * sizeof (data_type), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < input.size (); i++)
    if (excected_output[i] != output[i])
      throw std::runtime_error ("Error: unexpected value at " + std::to_string (i));

  cudaFree (device_input);
  cudaFree (device_result);
}

template <typename data_type>
void perform_single_value_warp_size_test (const data_type &magical_value)
{
  const int warpSize = 32;
  std::vector<data_type> iv32 (warpSize, data_type {});
  std::vector<data_type> ov32 (warpSize, magical_value);

  for (int lane = 0; lane < warpSize; lane++)
  {
    iv32[lane] = magical_value;
    expect_eq(iv32, ov32, warpSize);
    iv32[lane] = data_type {};
  }
}

template <typename data_type>
void perform_block_size_test (const data_type &magical_value)
{
  const int warpSize = 32;
  for (int multiplier = 1024 / warpSize; multiplier > 0; multiplier /= warpSize)
  {
    const int threads_count = warpSize * multiplier;
    std::vector<data_type> iv32 (threads_count, data_type {});
    std::vector<data_type> ov32 (threads_count, magical_value);

    for (int thread = 0; thread < threads_count; thread++)
    {
      iv32[thread] = magical_value;
      expect_eq(iv32, ov32, threads_count);
      iv32[thread] = data_type {};
    }
  }
}

template <typename data_type>
void perform_multiple_value_warp_size_test ()
{
  const int warpSize = 32;
  const int result = warpSize * (warpSize + 1) / 2;
  std::vector<data_type> iv32 (warpSize);
  std::vector<data_type> ov32 (warpSize, result);

  for (int i = 0; i < warpSize; i++)
    iv32[i] = i + 1;

  expect_eq(iv32, ov32, warpSize);
}

template <typename data_type>
void perform_multiple_value_block_size_test ()
{
  const int warpSize = 32;
  for (int multiplier = 1024 / warpSize; multiplier > 0; multiplier /= warpSize)
  {
    const int threads_count = warpSize * multiplier;

    const int result = threads_count * (threads_count + 1) / 2;
    std::vector<data_type> iv32 (threads_count);
    std::vector<data_type> ov32 (threads_count, result);

    for (int i = 0; i < threads_count; i++)
      iv32[i] = i + 1;

    expect_eq(iv32, ov32, threads_count);
  }
}

template <typename data_type>
void perform_tests (const data_type &magical_value)
{
  perform_single_value_warp_size_test(magical_value);
  perform_block_size_test(magical_value);

  perform_multiple_value_warp_size_test<data_type> ();
}

class user_type
{
public:
  unsigned long long int x {};
  unsigned long long int y {};

public:
  user_type () = default;
  __device__ __host__ user_type (int i) : x (i), y (0ull) {}
  __device__ __host__ user_type (unsigned long long int x_arg, unsigned long long int y_arg) : x (x_arg), y (y_arg) {}

  friend bool operator !=(const user_type &lhs, const user_type &rhs)
  {
    return lhs.x != rhs.x || lhs.y != rhs.y;
  }

  friend __device__ user_type operator+ (const user_type &lhs, const user_type &rhs)
  {
    return user_type (lhs.x + rhs.x, lhs.y + rhs.y);
  }
};

static_assert(std::is_trivially_copyable<user_type>::value, "User type should be trivially copyable");
static_assert(cuda::warp_reduce<user_type>::use_shared == true, "Default policy for warp reduce should use shared memory");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
static_assert(cuda::warp_reduce<int>::use_shared == false, "Default policy This type should use shfl");
static_assert(cuda::warp_reduce<long>::use_shared == false, "Default policy This type should use shfl");
static_assert(cuda::warp_reduce<long long>::use_shared == false, "Default policy This type should use shfl");
static_assert(cuda::warp_reduce<unsigned int>::use_shared == false, "Default policy This type should use shfl");
static_assert(cuda::warp_reduce<unsigned long>::use_shared == false, "Default policy This type should use shfl");
static_assert(cuda::warp_reduce<unsigned long long>::use_shared == false, "Default policy This type should use shfl");
static_assert(cuda::warp_reduce<float>::use_shared == false, "Default policy This type should use shfl");
static_assert(cuda::warp_reduce<double>::use_shared == false, "Default policy This type should use shfl");
#endif

class user_type_warp_shfl_reduce
{
public:
  user_type_warp_shfl_reduce () = default;

  template <typename _BinaryOperation>
  __device__ user_type __reduce_value (user_type __val, _BinaryOperation __binary_op)
  {
    for (int s = warpSize / 2; s > 0; s >>= 1) {
      unsigned long long new_x = __shfl_xor_sync(__FULL_WARP_MASK, __val.x, s, warpSize);
      unsigned long long new_y = __shfl_xor_sync(__FULL_WARP_MASK, __val.y, s, warpSize);

      __val = __binary_op(__val, user_type (new_x, new_y));
    }
    return __val;
  }

public:
  static constexpr bool use_shared = false;
};

void perform_benchmark ()
{
  cuda_benchmark::controller controller;

  user_type *input {};
  user_type *output {};

  const int warpSize = 32;

  std::vector<user_type> cpu_input (warpSize);
  cpu_input[11] = user_type (42);

  cudaMalloc (&input, warpSize * sizeof (user_type));
  cudaMalloc (&output, warpSize * sizeof (user_type));

  cudaMemcpy (input, cpu_input.data (), warpSize * sizeof (user_type), cudaMemcpyHostToDevice);

  controller.benchmark("warp reduce (user_type)", [=] __device__ (cuda_benchmark::state &state) {
    user_type thread_value = input[threadIdx.x];
    __shared__ char warp_workspace_data[32 * sizeof (user_type)];
    user_type *warp_workspace = reinterpret_cast<user_type*> (warp_workspace_data);

    for (auto _ : state)
    {
      cuda::warp_reduce<user_type> reduce (warp_workspace);
      thread_value = reduce (thread_value);
    }
  });

  controller.benchmark("warp reduce (user_type custom reduce policy)", [=] __device__ (cuda_benchmark::state &state) {
    user_type thread_value = input[threadIdx.x];
    for (auto _ : state)
    {
      cuda::warp_reduce<user_type, user_type_warp_shfl_reduce> reduce;
      thread_value = reduce (thread_value);
    }
  });

  cudaFree (output);
  cudaFree (input);
}

int main ()
{
#if 1
  perform_tests(42);
  perform_tests(42u);
  perform_tests(42ll);
  perform_tests(42u);
  perform_tests(42ul);
  perform_tests(42ull);
  perform_tests(user_type {4, 2});
#endif

  perform_benchmark();

  return 0;
}
