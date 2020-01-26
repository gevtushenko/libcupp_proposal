#include <type_traits>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/numeric"

template <cuda::thread_scope scope, typename data_type>
__global__ void perform_reduce (const data_type *array, unsigned int n, data_type *result)
{
  int value = cuda::reduce<scope>(array, array + n, data_type {}, [] __device__ (const data_type &a, const data_type &b) { return a + b; });

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
  perform_reduce<cuda::thread_scope_warp><<<blocks_count, threads_per_block>>>(device_input, input.size (), device_result);

  std::vector<data_type> output (input.size (), data_type {});
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

class user_type
{
  unsigned long long int x {};
  unsigned long long int y {};
public:
  user_type () = default;
  user_type (unsigned long long int x_arg, unsigned long long int y_arg) : x (x_arg), y (y_arg) {}

  friend __device__ user_type operator+ (const user_type &lhs, const user_type &rhs)
  {
    return user_type (lhs.x + rhs.x, lhs.y + rhs.y);
  }
};

static_assert(std::is_trivially_copyable<user_type>::value);

int main ()
{
  perform_single_value_warp_size_test(42);
  // perform_single_value_warp_size_test(user_type {4, 2});

  return 0;
}
