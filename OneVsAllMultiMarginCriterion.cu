#include "utils.h"

#define MULTIMARGIN_THREADS 128

__global__ void cunn_OneVsAllMultiMarginCriterion_updateOutput_kernel(float *output, float *input, float *target, int nframe, int dim, int sizeaverage, float positiveWeight)
{
  __shared__ float buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k;
  int target_k = ((int)target[k])-1;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for(int i = i_start; i < i_end; i += i_step)
  {

    float y = (i==target_k) ? 1.0 : -1.0;
    float z = 1 - input_k[i]*y;         
    if(z > 0){
        float weight = (i==target_k) ? positiveWeight : 1.0;
        buffer[threadIdx.x] += z;
    }
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum = 0;
    for (int i=0; i<blockDim.x; i++)
      sum += buffer[i];

    if(sizeaverage)
      *output_k = sum/dim;
    else
      *output_k = sum;
  }
}

__global__ void cunn_OneVsAllMultiMarginCriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, int nframe, int dim, int sizeaverage, float positiveWeight)
{
  __shared__ float buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *gradInput_k = gradInput + k*dim;
  int target_k = ((int)target[k])-1;
  float g = (sizeaverage ? 1./((float)dim) : 1.);

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float y = (i==target_k) ? 1.0 : -1.0;
    float z = 1 - input_k[i]*y;

    if(z > 0)
    {
      float weight = (i==target_k) ? positiveWeight : 1.0;
      float h =  -y*g*weight;
      gradInput_k[i] = h;
    }
    else
      gradInput_k[i] = 0;
    }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float gradInput_target_k = 0;
    for (int i=0; i<blockDim.x; i++)
      gradInput_target_k += buffer[i];
    gradInput_k[target_k] = gradInput_target_k;
  }
}

static int cunn_OneVsAllMultiMarginCriterion_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  float positiveWeight = luaT_getfieldchecknumber(L, 1, "positiveWeight");


  input = THCudaTensor_newContiguous(state, input);

  if(input->nDimension == 1)
  {
    float target_ = luaL_checknumber(L, 3);
    THCudaStorage *target = THCudaStorage_newWithSize(state, 1);
    THCudaStorage *output = THCudaStorage_newWithSize(state, 1);
    dim3 blocks(1);
    dim3 threads(MULTIMARGIN_THREADS);

    THCudaStorage_fill(state, target, target_);

    cunn_OneVsAllMultiMarginCriterion_updateOutput_kernel <<<blocks,threads>>>(output->data,
                                                                      THCudaTensor_data(state, input),
                                                                      target->data,
                                                                      1, input->size[0],
                                                                      sizeaverage,positiveWeight);
    
    lua_pushnumber(L, THCudaStorage_get(state, output, 0));

    THCudaStorage_free(state, output);
    THCudaStorage_free(state, target);
  }
  else if(input->nDimension == 2)
  {
    THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    THCudaTensor *output = THCudaTensor_newWithSize1d(state, input->size[0]);
    dim3 blocks(input->size[0]);
    dim3 threads(MULTIMARGIN_THREADS);
    cunn_OneVsAllMultiMarginCriterion_updateOutput_kernel <<<blocks,threads>>>(THCudaTensor_data(state, output),
                                                                      THCudaTensor_data(state, input),
                                                                      THCudaTensor_data(state, target),
                                                                      input->size[0], input->size[1],
                                                                      sizeaverage,positiveWeight);

    lua_pushnumber(L, THCudaTensor_sumall(state, output));
    THCudaTensor_free(state, output);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  THCudaTensor_free(state, input);
  return 1;
}

static int cunn_OneVsAllMultiMarginCriterion_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  float positiveWeight = luaT_getfieldchecknumber(L, 1, "positiveWeight");

  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor_resizeAs(state, gradInput, input);

  if(gradInput->nDimension == 1)
  {
    float target_ = luaL_checknumber(L, 3);
    THCudaTensor *target = THCudaTensor_newWithSize1d(state, 1);
    dim3 blocks(1);
    dim3 threads(LOGSOFTMAX_THREADS);

    THCudaTensor_fill(state, target, target_);

    cunn_OneVsAllMultiMarginCriterion_updateGradInput_kernel <<<blocks,threads>>>(THCudaTensor_data(state, gradInput),
                                                                         THCudaTensor_data(state, input),
                                                                         THCudaTensor_data(state, target),
                                                                         1, gradInput->size[0],
                                                                         sizeaverage,positiveWeight);
  
    THCudaTensor_free(state, target);
  }
  else if(gradInput->nDimension == 2)
  {
    THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
    dim3 blocks(gradInput->size[0]);
    dim3 threads(LOGSOFTMAX_THREADS);

    cunn_OneVsAllMultiMarginCriterion_updateGradInput_kernel <<<blocks,threads>>>(THCudaTensor_data(state, gradInput),
                                                                         THCudaTensor_data(state, input),
                                                                         THCudaTensor_data(state, target),
                                                                         gradInput->size[0], gradInput->size[1],
                                                                         sizeaverage,positiveWeight);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
}

static const struct luaL_Reg cunn_OneVsAllMultiMarginCriterion__ [] = {
  {"OneVsAllMultiMarginCriterion_updateOutput", cunn_OneVsAllMultiMarginCriterion_updateOutput},
  {"OneVsAllMultiMarginCriterion_updateGradInput", cunn_OneVsAllMultiMarginCriterion_updateGradInput},
  {NULL, NULL}
};

static void cunn_OneVsAllMultiMarginCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_OneVsAllMultiMarginCriterion__, "nn");
  lua_pop(L,1);
}
