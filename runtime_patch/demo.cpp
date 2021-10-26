//
// Created by mlevental on 10/25/21.
//

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace std;

static int NUM_ELEMENTS;
static char *BACKEND;

vector<int> add_cpu(vector<int> a, int a_size, vector<int> b, int b_size) {
  if (a_size != b_size)
    throw invalid_argument("mismatched sizes");
  vector<int> c(a_size, 0);
  for (int i = 0; i < a_size; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

struct MaxTensor {
  vector<int> ls;
  int size;

  MaxTensor(vector<int> ls) {
    this->size = ls.size();
    this->ls = std::move(ls);
  }

  MaxTensor operator+(const MaxTensor &rhs) const {
    if (string(BACKEND) == "cpu") {
      return {add_cpu(this->ls, this->size, rhs.ls, rhs.size)};
    } else {
      throw invalid_argument("unimplemented");
    }
  }
};

MaxTensor relu(MaxTensor a) {
  vector<int> c(a.size, 0);
  for (int i = 0; i < a.size; ++i) {
    c[i] = max(0, a.ls[i]);
  }
  return c;
}

MaxTensor add_relu_cpu(MaxTensor a, MaxTensor b) {
  if (a.size != b.size)
    throw invalid_argument("mismatched sizes");
  vector<int> c(a.size, 0);
  int tmp;
  for (int i = 0; i < a.size; ++i) {
    tmp = a.ls[i] + b.ls[i];
    c[i] = max(0, tmp);
  }
  return c;
}
#include <iostream>
int main() {
  std::cout << "hello";
  NUM_ELEMENTS = atoi(getenv("NUM_ELEMENTS"));
  BACKEND = getenv("BACKEND");

  std::cout << BACKEND << ", " << NUM_ELEMENTS << "\n";

  MaxTensor t1(vector<int>(NUM_ELEMENTS, 0));
  MaxTensor t2(vector<int>(NUM_ELEMENTS, 0));

  auto t3 = t1 + t2;
  auto t4 = relu(t3);
}

#include "utils.h"

__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
                                  unsigned char *const greyImage, int numRows,
                                  int numCols) {

  size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  size_t c = blockIdx.y * blockDim.y + threadIdx.y;

  if (r < numRows && c < numCols) {
    size_t offset = r * numCols + c;
    uchar4 rgba = rgbaImage[offset];
    float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
    greyImage[offset] = channelSum;
  }
}

void add(const uchar *const h_rgbaImage,
                            uchar4 *const d_rgbaImage,
                            unsigned char *const d_greyImage, size_t numRows,
                            size_t numCols) {
  // You must fill in the correct sizes for the blockSize and gridSize
  // currently only one block with one thread is being launched

  const size_t tileSize = 16;
  const dim3 gridSize(1 + numRows / tileSize, 1 + numCols / tileSize, 1);
  const dim3 blockSize(tileSize, tileSize, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows,
                                             numCols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}