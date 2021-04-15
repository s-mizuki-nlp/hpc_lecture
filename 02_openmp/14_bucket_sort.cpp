#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

// cumulative sum parallelized with prefix sum technique.
std::vector<int> parallel_prefix_sum(std::vector<int>& vec) { 
  int N = vec.size();
  std::vector<int> dummy(N,0);
  std::vector<int> ret = vec;

#pragma omp parallel
  for(int j=1; j<N; j<<=1) {
#pragma omp for
    for(int i=0; i<N; i++)
      dummy[i] = ret[i];
#pragma omp for
    for(int i=j; i<N; i++)
      ret[i] += dummy[i-j];
  }

  return ret;
}

int main(int argc, char *argv[]) {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
#pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++)
#pragma omp atomic update
    bucket[key[i]]++;

  // discard offset variable, instead we use bucket_cumsum.
  // std::vector<int> offset(range,0);
  // bucket_cumsum[t] = \sum_{u=0}^{t} bucket[u]
  std::vector<int> bucket_cumsum = parallel_prefix_sum(bucket);

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    // int j = offset[i];
    int j = i == 0 ? 0 : bucket_cumsum[i-1];
    for (int k = bucket[i]; k>0; k--)
      key[j++] = i;
  }

  for (int i=0; i<n; i++) {
    if (i<n-1) assert(key[i] <= key[i+1]);
    printf("%d ",key[i]);
  }
  printf("\ncompleted.");
}
