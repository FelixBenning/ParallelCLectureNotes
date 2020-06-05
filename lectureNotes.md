# Command line

- `lscpu` information about `native` CPU flags
- `top` (`htop`) provides (detailed) information about running processes
- `kill -9 pid` where `pid` is the process identification, kills this process
- `ssh username@hostIPAdress` log into a remote server
- `scp fromFileLocal.txt username@hostIPAdress:toFile.txt` copy over ssh

# Compiler

## Precompiler
https://gcc.gnu.org/onlinedocs/cpp/Macros.html#Macros

Macros are expanded before substitution (only one substitution per Macro)

```c
#define A B //only one substitution
#define B A
A //expansion -> B -> A -> stop
```

``` c
#define f(a) //no space between f and (
f(a,b) //error
f((a,b)) //ok
```

#### Stringizing operator (#)

```c
#define A(x) #x
A(something) // result: "something"
```

#### Token-pasting operator (##)

``` c
#define A XX
#define B(Y) ##Y
#define C(Y) Y
B(A) //result A
C(A) //result XX
    
#define aus_alt_mach_neu 42
#define AUS(X) aus_alt_##X ## _neu
AUS(mach) // ergibt 42
```

#### Variable Number of Arguments

``` c
#define P(x, ...) printf(x, __VA_ARGS__);
printf(stderr, format __VA_OPT__(,) __VA_ARGS__)
```

`__VA_OPT__(,)` inserts a comma iff `__VA_ARGS__` is not empty 

#### Commands

- `#include "local.h"` `#include <global.h>` 
- `#define`, `#undef`
- `#error string`, `#warning string`
- `#if`, `#elif`, `#else`, `#endif` (only whole numbers, macros and strings; operators: - * / << >> && || < > == != ! <= >=; values unequal to zero are truthy)
  - `#if defined` (alias `#ifdef`), `#if !defined` (alias `#ifndef`)
  - `#if UNDEFINED_VAR > 0` undefined equation evaluate to false

## Constants
Useful predefined macros:
https://gcc.gnu.org/onlinedocs/cpp/Standard-Predefined-Macros.html#Standard-Predefined-Macros

- `__cplusplus` is C++ compilation?
- `__FILE__`, `__LINE__`
- `__AVX__`, `__AVX2__` SIMD
- `_OPENMP` is OMP compilerflag set?
- `_WIN32` is windows?
- `__GNUC__`, `GNUC_MINOR__`, `__GNUC_PATCHLEVEL__` compiler version
- `__CHAR_BIT__`, `__SIZEOF_LONG__`,`__SIZEOF_POINTER__`, ... 
- `__BYTE_ORDER__ ==__ORDER_LITTLE_ENDIAN__` 

## Compiler Flags

### gcc

#### Additional Warnings

Also cf. [gcc Warning Options](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html)

- `-Wall` : many warnings, including
  - `-Wformat`: special checks for `printf`/`scanf` additional options (not in `-Wall`):
    - `-Wformat-security` : warning if the format string in a `printf`/`scanf` function is not a string literal (cf. code injection)
    - `-Wformat-y2k` : also warn about `strftime` formats that may yield only a two-digit year
    - ...
  - `-Wswitch` warnings for the switch statement (e.g. lacks of cases, case labels outside the enumeration, ...), additional options (not in `-Wall`):
    - `-Wswitch-default`: default has to exist in switch statement
  - `-Wuninitialized` : check if variable is used although not initialized
  - ...
  - Ignore certain warnings:
    - `-Wno-unused-variable` ,`-Wno-unused-function`
    - `-Wno-parentheses` : stops forcing parentheses in situations like this X && YY || Z.
- `-Wextra` (alias `-W`) : additional warnings
- `-pedantic` : ISO C vs ISO C++ problems
- `-Wshadow ` : local variable has same name as global

#### Architecture flags

cf. [gcc x86 Options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html)

- `-march=cpu-type`  compile for **m**achine **arch**itecture of `cpu-type`, e.g. `native` , `x86-64`,`tigerlake`,... Sets all the flags available for `cpu-type` and thus will not run on cpus without these flags (usually runs on newer than `cpu-type` cpus), e.g.
  - `-mavx`, `-mavx2`,... enables the use of AVX, AVX2,... instructions, with corresponding `-mno-`option for deactivation 
- `-mtune=cpu-type` tune the generated code for `cpu-type`, but will still run on the default machine the compiler is configured for (in contrast to `march`)

#### OMP flags

- `-pthread` add support for multithreading using the POSIX threads library (links library and sets flags)
- `-fopenmp` enables OpenMP directive `#pragma omp` (automatically links the pthread library)
- `-Xpreprocessor -fopenmp -lomp` Apple shit

#### Other flags

- `-std=c11` use C standard C11 (published in 2011)
- `-O0`, ..., `-O3` optimization level (`-O3` is aggressive)
- `-funroll-loops` might provide speed up in certain situations and is not included in `-O3`
- `-g0`(default), `g1`, `-g`, `-g3` levels of debugging/symbol information retained (e.g. variable names), for use in a debugger
- `-Dname` predefine `name` as a macro with definition 1 
  - `-DNDEBUG` disable assertions (i.e. `#define NDEBUG 1` )
- `-pipe` use pipes instead of temporary files during compilation
- `-lm` links math library (i.e. provides the implementation for the headers included with `#include <math.h>`)
- `-E` stop after the preprocessing stage; the output is preprocessed source code
- `-o file` place the primary output in `file` 
- `-Idir` search `dir` for header files which might be included in some source file
- `-fPIC` position independent code (for shared libraries), i.e. relative memory addresses instead of absolute memory addresses

#### nvcc

cf. [nvcc compiler options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-compiler-options)

- `--compiler-options options`, (alias`-Xcompiler`) passes `options` to the c compiler called by nvcc to compile the c/c++ code.

# Parallel Computing

## SIMD

(**S**ingle **I**nstruction **M**ulti **D**ata - vectorized calculation) see [Intel Intrinsics Guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide) 

### Loop unrolling

Loop unrolling can help the compiler identify loops with independent instructions which might be vectorized. 

```c
for (int i=0; i<n; i+=4) {
	z[i] += x[i] * y[i];
	z[i+1] += x[i+1] * y[i+1];
	z[i+2] += x[i+2] * y[i+2];
	z[i+3] += x[i+3] * y[i+3];
}
```

Not necessary nor sufficient for the compiler to try, but less error prone than intrinsics.

### [Using Intel Intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide)

[LMU cheatsheet](https://db.in.tum.de/~finis/x86-intrin-cheatsheet-v2.1.pdf)

make sure that your computer can use these types and operations (cf. [Command line](#Command Line), [Compiler Constants](#constants), [Architecture flags](#architecture-flags))

```c
#include <immintrin.h>
```

#### Types

`__m[bits][type]`

Bitsizes: AVX/AVX2: `128`, `256`, AVX-512: `512`

- `__m256` 256 bits (usually 8x32 `float`)
- `__m256d` 256 bits (usually 4x64 `double`)
- `__m256i` 256 bits (usually 4x64 `uint64_t`, 8 `int32_t`, 16 `int16_t`, ...)

<small> with fixed width integer types from `stdint.h` (the superset `inttypes.h` enables `printf`/`scanf`)</small>

##### Warning!

```c
double* d_address = malloc(4*sizeof(double));
__m256d* cast_address = (__256d*) d_address;
```

might cause unexplained crashes due to memory alignment problems. Necessary:

```c
type_address%sizeof(type)==0 //address is in bytes, sizeof returns bytes!
```

#### Functions

`_mm[bits]_fct_name`

- Data Type Suffixes:
  - `pd` **p**acked **d**ouble
  - `ps` **p**acked **s**ingle
  - `si256` **s**ingle **i**nteger of size **256** bits
  - `epi256` **p**acked **i**nteger of size **256** bits
  - `epu256` **p**acked **u**nsigned integer of size **256** bits

- Load: 

  - ```c
    __m256d _mm256_load_pd(double const *mem_addr)
    ```

    fetches `__m256d` ) from memory (needs to be aligned!)

  - ```c
    __m256d _mm256_loadu_pd(double const *mem_addr)
    ```

    fetches `__m256d`  from memory (**u**naligned)

  - ```c
    __m256 _mm256_load[u]_ps(float const *mem_addr)
    ```

  - ```c
    __m256i _mm256_load[u]_si256(__m256i const *mem_addr)
    ```

- Store: cf. Load (search for `store_` & `storeu_`)
- Arithmetic (search intrinsics guide):
  -  `add`, `mul`, `sub`, `div`
  - `min`, `max`, `avg`, `sqrt`, `abs`, `ceil`, `floor`
  - `hadd` horizontal add

#### Union struct

```c
union uni256 {
    __m256i i;
    __m256d d;
    double d8[4];
};
```

allows for easy conversion (`uni256.i`, ...)

## OMP

(**O**pen **M**ulti-**P**rocessing) see [Programming Prallel Computers](http://ppc.cs.aalto.fi/) and [Microsoft OpenMP Documentation](https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-library-reference?view=vs-2019)

```c
#include <omp.h>
```

syntax:

``` c
#pragma omp [directive] [clauses]
```

### [Directives](https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-directives?view=vs-2019)

#### parallel

```c
#pragma omp parallel [clauses: if,private,firstprivate,default,shared,copyin,reduction,num_threads]
{
    code_block
}
```

can be used together with for and sections

#### for

```c
#pragma omp parallel for [clauses: private,firstprivate,lastprivate,reduction,ordered,schedule,nowait]
for (int i = 0; i<n; i++) {...}
```

or inside a parallel block

```c
#pragma omp parallel [clauses]
{
    ...
    #pragma omp for [clauses]
    for (int i =0; i<n; i++) {...}
}
```

#### sections

```c
#pragma omp parallel sections [clauses: private,firstprivate,lastprivate,reduction,nowait] 
{
	#pragma omp section
    {
        code_block_section1
    }
    #pragma omp section
    {
        code_block_section2
    }
    ...
}
```

executes the sections in parallel. `sections` can also be used inside a parallel block cf. `for`.

#### Sync directives

##### barrier

```c
#pragma omp barrier //threads wait for all other threads here
```

##### single

```c
#pragma omp single [clauses:private,firstprivate,copyprivate,nowait]
{
    code_block //executed by a single thread
}
```

##### master

like single, no clauses supported, executed by thread 0 (master thread).

##### ordered

specifies that code under a parallelized `for` loop should be executed like a sequential loop, no clauses, `for` directive needs the `ordered` clause to enable this feature

```c
#pragma omp parallel for ordered
for(int i=0; i<n; i++){
    //do something
    #pragma omp ordered
    printf("output %d: %f",i, result);
}
```

##### atomic

Protect a memory location against multiple writes at once, forces other threads to wait

```c
#pragma omp atomic
var binop= expr; //or var++, etc
```

`binop` is one of  `+`, `*`, `-`, `/`, `&`, `^`, `|`, `<<`, or `>>` (not overloaded)

##### critical

```c
#pragma omp critical [(name:optional)]
{
    code_block
}
```

code is only to be executed by one thread at a time (in general slower than atomic)

##### flush

```c
#pragma omp flush [(var:optional)]
```

flushes the cache (syncs the variables) for all `var`, if `var` is not specified, flushes all.

### [Clauses](https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-clauses?view=vs-2019)

#### general clauses

##### nowait

overrides the barrier implicit in a directive

##### num_threads(`num`)

set the number of threads in a thread team to `num`

##### if(`expr`)

only execute the code (in parallel region) in parallel if `expr`, otherwise single thread

##### schedule

```c
#pragma omp parallel for schedule(type[,chunk_size])
```

`type` can be:

- `static`: chunks are assigned to the threads in the order of the thread number. `chunk_size` defaults to (approximately) `iterations/num_threads`
- `dynamic`:  chunks are assigned to a thread which is waiting for an assignment. `chunk_size` defaults to 1.
- `guided`: threads are assigned chunks with decreasing sizes. For a `chunk_size` of 1, the size of each chunk is approximately `iterations/num_threads`. For `chunk_size=k` with `k>1`, the sizes decrease almost exponentially in k. `chunk_size` defaults to 1
- `runtime`: decision is deferred until runtime. Can be chosen by setting the environment variable `OMP_SCHEDULE`. If not set, the resulting schedule is implementation defined. In this case `chunk_size` *must not* be set. 

#### data-sharing clauses

##### reduction

- default implementation for basic arithmetic variables and operators (+,*,...)

  ``` c
  long double result=0.0;
  #pragma omp parallel for reduction (+:result)
  for (int i =0; i<n; i++) result++;
  
  //result==n
  ```

- custom implementation using:

  ``` c
  add_int(int a, int b){
      return a+b;
  }
  #pragma omp declare reduction (\
      /*identifier*/ r_name :\
      /*typelist*/ int :\
      /*combiner*/omp_out=add_vec(omp_out, omp_in)\
  ) initializer(omp_priv = 1)
  
  int result = 10;
  #pragma omp parallel for reduction(r_name:result)
  for(int i = 0 ; i<n; i++) /*do nothing*/;
  
  //result == num_threads * 1(or omp_priv) + 10
  ```

##### private(`var`'s)

specifies that each thread should have its own instance of `var`, the original value is *not* copied and `var` is uninitialized when entering the parallel block, and is uninitialized after the parallel block

##### firstprivate(`var`'s)

like private, but copies the original value of `var` into the thread private versions.

##### lastprivate(`var`'s)

the `var` variable after the parallel block is set equal to the private version of the thread which executes the last iteration (for directive) or last section (sections directive).

##### shared(`var`'s)

specifies that (the memory location of) `var` is shared across all threads

##### default(either `shared` or `none`)

specifies that `var` used inside a parallel region from outside the parallel region default to `shared` (default behavior) or cause a compiler error if unspecified (`none` behavior).

##### copyin(`var`'s)

provides a mechanism to assign the same value to `threadprivate` variables for each thread in the team executing parallel regions.

##### copyprivate(`var`'s)

applies to the single directive and copies the `var` from this single thread into private versions

### [OpenMP Functions](https://docs.microsoft.com/en-us/cpp/parallel/openmp/reference/openmp-functions?view=vs-2019)

#### Get Settings

``` c
omp_get_num_procs(); //Processors (Hardware)
omp_get_num_threads(); //current number of threads in parallel region
omp_get_max_threads(); //number of threads if a parallel region without num_threads were defined at this point
omp_in_parallel(); //called from within a parallel region?
omp_get_dynamic(); //can the number of threads in the upcoming parallel region be adjusted by the run time?
```

#### Set Settings

```c
omp_set_num_threads(); //Set num of threads in upcoming parallel regions, unless overridden by a num_threads clause.
omp_set_dynamic(); //indicate that the number of threads in the upcoming parallel region can be adjusted by the run time
```



## Cuda

[Cuda by Example](https://github.com/jiekebo/CUDA-By-Example/blob/master/common/book.h)

### GPU Architecture

Abstractions:

- **Threads** are grouped into **Blocks** which are assigned to a **S**treaming **M**ultiprocessors (SM) and can not be reassigned
- **warp** sized (usually 32) chunks of these blocks are executed in a SIMD like fashing inside the SM in no determined order. Blocksizes should therefore be multiples of warpsize. But Cuda deals with masking in case of partially filled warps, or branching points (`if`, `switch`, etc.), ignorning this fact only costs performance
- Halting execution, switching to another block/warp, and continuing execution within an SM is very cheap.

Hardware:

- SM have multiple **warp scheduler** with associated computing capability, with *shared register* and *instruction cache* (**L0**)

- Memory levels:

  | Level                |     Size | Latency (GPU cycles) |    Bandwidth |
  | :------------------- | -------: | -------------------: | -----------: |
  | SM level (**L1**)    | 16-48 kB |                   80 |              |
  | GPU level (**L2**)   |   2-4 MB |              200-300 | 500-1000GB/s |
  | Graphic Card (DRAM*) |  4-16 GB |              200-300 |  250-500GB/s |

  *: **D**ynamic **R**andom **A**ccess **M**emory

### Basic Usage

File endings are `.cu` (instead of `.c`,`.cc`, or `.cpp`) and it is necessary to use the Nvidia compiler `nvcc`, which delegates some tasks to a normal C compiler. So both needs to be installed.

#### Decorators

- `__host__` compile as normal CPU function (equivalent to no decorator)
- `__device__` compile as GPU function
- `__global__` compile as **kernel** (a GPU function callable by the CPU)

Example:

``` c
__host__ __device__ int min(int a, int b){
    return a<b ? a : b;
}
```

here the compiler produces a version of `min` callable by the cpu and another version callable by the gpu.

#### Managing Devices

[Cuda Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE)

- ```c
  int count;
  cudaGetDeviceCount(&count);
  ```

  provides the number of Nvidia GPUs with compute capability of at least 1.0 (devices are enumerated from 0,...,n-1)

- ```c
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, d_num)
  ```

  fills the [`cudaDeviceProp` struct](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html) with the properties of device `d_num`

- ```c
  __host__ __device__ cudaError_t cudaGetDevice(int *d_num);
  __host__ cudaError_t cudaSetDevice(int d_num);
  ```

  get/set the current device context (i.e. the device which kernel calls are passed to)

#### Calling a Kernel

```c
const BLOCKS = n;
const BLOCKSIZE = k*32;

__global__ void mult(float *gpu_a, float *gpu_b, float *gpu_c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N) gpu_c[idx] = gpu_a[idx] * gpu_b[idx];
}

int main(void) {
    // data management providing gpu pointers gpu_a,...gpu_c
    mult <<<BLOCKS, BLOCKSIZE>>> (gpu_a, gpu_b, gpu_c);
    //use result gpu_c
}
```

Calling a kernel is asynchronous and the CPU thread continuous without waiting for the result of the kernel. But some of the memory management commands have synchronization built in. See [Concurrency](#concurrency) for further details.

#### Data Management

##### Global Memory

```c
float* cpu_pointer = (float*) malloc(10*sizeof(float));
//fill with data
//========================== GPU ==============================
float* gpu_pointer;
cudaMalloc((void**) &gpu_pointer, 10*sizeof(float));

cudaMemcpy(gpu_pointer, cpu_pointer, cudaMemcpyHostToDevice);
//do stuff on the gpu with gpu_pointer
cudaMemcpy(cpu_pointer, gpu_pointer, cudaMemcpyDeviceToHost);

cudaFree(gpu_pointer);
//=============================================================
//use data from GPU
free(cpu_pointer);
```

`cudaMalloc`, `cudaMemcpy`, `cudaFree` allocate, copy and free memory on the device DRAM. This memory is also referred to as global memory, which can be misleading as it is not shared with the CPU.

It is also possible to directly access CPU RAM from the device utilizing [`cudaMallocHost`](http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/docs/online/group__CUDART__MEMORY_g9f93d9600f4504e0d637ceb43c91ebad.html#g9f93d9600f4504e0d637ceb43c91ebad)

##### Shared Memory

[Reference 1: Cuda Succinctly](https://www.syncfusion.com/ebooks/cuda/shared-memory), [Reference 2: Nvidia Devblog](https://devblogs.nvidia.com/using-shared-memory-cuda-cc/)

It is also possible to allocate Shared Memory in the L1 cache (SM wide). This memory is shared by all threads within a block. Recall: Blocks are assigned and stay with one SM.

- Static Allocation

  ```c
  __global__ void kernel(){
      __shared__ int num;
      __shared__ float farr[10];
  }
  ```

- Dynamic Allocation

  ```c
  __global__ void kernel(){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      extern __shared__ int num[];
      //do something with num[idx]
  }
  ```

  ```c
  kernel <<< blocks, blocksize, blocksize*sizeof(int)>>>()
  ```

  the third argument of the  `<<<>>>` parameters specifies the size of *the* `extern __shared__` variable. For this reason multiple, dynamically allocated arrays require a split of this one array:

  ``` c
  __global__ void kernel(int isize, int fsize){
      extern __shared__ int arr[];
      int *iarr = arr;
      float *farr = (float*) arr + isize;
  }
  ```

Due to the dual functionality of the L1 memory as cache and shared memory, it needs to be split. [`cudaFuncSetCacheConfig`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6699ca1943ac2655effa0d571b2f4f15) influences the sizes of this split. 

#### Intrinsic functions

[Cuda Maths API Dokumentation](https://docs.nvidia.com/cuda/cuda-math-api/index.html) 

*Note*: there is a difference between Single/Double Precision *Mathematical Functions* which can also be used in`__host__` code and Single/Double Precision *Intrinsics*. 

| Mathematical Function/Operation               | Intrinsics                            |
| --------------------------------------------- | ------------------------------------- |
| `x/y`                                         | `__fdividef(x,y)`                     |
| `sinf(x)`, `cosf(x)`, `tanf(x)`               | `__sinf(x)`, `__cosf(x)`, `__tanf(x)` |
| `expf(x)`, `logf(x)`, `log2f(x)`, `powf(x,y)` | `__expf(x)`, ... , `__powf(x,y)`      |

Using the compiler option `-use_fast_math` converts all functions to intrinsics cf. [Programming Guide Appdx E.2: Intrinsic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions). The advantage is speed, the drawback is precision and missing special case handling. It is thus advised to avoid the compiler option and explicitly convert functions where applicable.

### Concurrency

#### Synchronization

##### CPU/GPU Synchronization

Calls to kernels are queued up in order (cf. [Concurrent Kernel Execution](#Concurrent-Kernel-Execution) for a generalization), but the CPU continuous asynchronously. In order to force the CPU to wait, you could use

```c
__host__ __device__ cudaError_t cudaDeviceSynchronize ( void );
```

This synchronization is built into `cudaMemcpy`, which is why there exists a variant called `cudaMemcpyAsync` cf. [Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization)

##### Thread Synchronization

In order to synchronize threads in a block, it is possible to introduce a barrier with

``` c
__syncthreads()
```

Useful when dealing with shared memory. Although Atomics might be more suited for this purpose.

#### Atomics

[Atomic Functions Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions) An operation "is atomic in the sense that it is guaranteed to be performed without interference from other threads." Atomic operations usually read a certain memory location, perform an operation, store the result at the memory location and return the old value. Examples:

- ```c
  int atomicAdd(int* address, int val);
  ```

  also `unsigned int`, `float`,`double`, similarly: `atomicSub`

- ```c
  int atomic Exch(int* address, int val);
  ```

  returns the value at address and stores val into the address

- ```c
  int atomicMin(int* address, int val);
  ```

  similarly `atomicMax`

#### Concurrent Kernel Execution

[Stream&Concurrency Webinar](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf), [Nvidia Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)

##### [Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)

see [Stream Management API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)

If not further specified, kernel calls are queued up in the [Default Stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#default-stream), it is possible to create different queues (streams), which can execute concurrently:

```c
cudaStream_t stream[2];
cudaStreamCreate(&stream[0]);
cudaStreamCreate(&stream[1]);

kernel <<< blocks, blocksize, shared_mem_size, &stream[0]>>> ();
kernel <<< blocks, blocksize, shared_mem_size, &stream[1]>>> (); 
```

Such a stream also needs to be provided to [`cudaMemcpyAsync`](http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/docs/online/group__CUDART__MEMORY_ge4366f68c6fa8c85141448f187d2aa13.html). 

*Warning*: Assigning a kernel call to the default stream either implicitly or by passing `0` or `NULL`  blocks ALL following kernel calls from executing before this call is finished, even calls to other streams. 

It is also possible to synchronize the CPU with a certain GPU stream:

```c
__host__ cudaError_t cudaStreamSynchronize(cudaStream_t stream)
```

##### [Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)

see [Event Management API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)

It is possible to create event objects:

```c
__host__ cudaError_t cudaEventCreate ( cudaEvent_t* event );
__host__  __device__ cudaError_t cudaEventDestroy ( cudaEvent_t event );
```

which can then be used to record events:

```c
__host__  __device__ cudaError_t cudaEventRecord (cudaEvent_t event, cudaStream_t stream = 0 );
```

Recording an event for `stream1` means, the event is put into the `stream1` queue, and is completed as soon as the tasks in front of the event complete. The completions is also saved as a time stamp which enables the use of events for 

```c
__host__ cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )
```

but events can also be used for synchronization of the CPU

```c
__host__ cudaError_t cudaEventSynchronize ( cudaEvent_t event )
```

which blocks until this event is completed, or for synchronization between streams:

```c
__host__  __device__ cudaError_t cudaStreamWaitEvent ( cudaStream_t stream, cudaEvent_t event, unsigned int  flags ) 
```

this queues a barrier into `stream`, which clears as soon as `event` completes. 

###### Example:

```c 
cudaEvent_t event1;
cudaEventCreate(event1);
myKernel <<< b, bs, sh_mem, stream1>>> (); //queuePos#1 in stream1
cudaEventRecord(event1, stream1); //queuePos#2 in stream1
cudaStreamWaitEvent(stream2, event1, 0); //queuePos#1 in stream2
myKernel2<<<b,bs,sh_mem, stream2>>>(); //queuePos#2 in stream2
```

`myKernel2` is executed if the `queuePos#1` clears in front of it. This only happens if `event1` is completed. Which requires `queuePos#1` of `stream1` to clear such that the `event1` at `queuePos#2` is completed.

### Blocksize



## General Advice

- **Local Variables**: If Variables are as local as possible they can be optimized by the compiler to become temporary cache/register entries. This removes a lot of costly memory allocation.

- **Array of Structs vs Struct of Arrays**: 

  - CPUs fetch adjacent memory entries, which means that arrays of structs can make sense, if the entire struct is used, every time it is fetched from memory. If the first entry of every struct in the array is used, it might make sense to reformat it into a struct of arrays. Which means that the n-th entries of adjacent structs are adjacent.

  - Since GPUs execute every step as a warp of threads, not even immediate usage of all entries of a struct justify an array of structs. Imagine:

    ``` c
    struct twoElmts {int first; int second;};
    
    __global__ myKernel(struct twoElmts* ptr){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int myElmt1 = ptr[idx].first; // 1) 
        int myElmt2 = ptr[idx].second; // 2)
    }
    ```

    An array of `twoElmts` structs, is an alternating list of first and second elements. At `1)`, every thread in the warp, needs the first element. Thus 32 *even* elements of the list have to be fetched. At `2)` 32 *odd* elements of the list are fetched. As a result a continuous stretch of 64 integer sized memory is fetched *twice*. 

- **Inline Functions**: If you are starting to section parts of your code with comments, explaining what these sections do, then these sections probably should be functions. Functions can be given descriptive names which can essentially act as a section title. This "collapses" the sections to a table of contents, where the "headings" (i.e. function names) can be further inspected by reading the function definition.  If you are worried that these function calls might have an impact on performance, you can give the compiler a hint (`inline`), that it might be desirable, if the function definition is simply copied in place of the function call.

- **Only use Macros if absolutely necessary**: Precompiler Macros are incredible powerful as it literally allows you to procedurally write code. For this very same reason its use should be avoided, since it also allows you to horribly break things. People like to joke about things like

  ``` c
  #define false true
  #define if while
  ...
  ```

  and while these might seem extreme. The precompiler can help you write code which is incredibly difficult to read and debug.

  Consider for example:

  ```c
  #define THREADS pow(2,7)
  #define THREADS_PER_BLOCK 100
  #define BLOCKS (1 + (THREADS - 1) / THREADS_PER_BLOCK)
  ```

  `BLOCKs` is `ceil(THREADS/THREADS_PER_BLOCK)`, right? No! While everything looks like integers, `pow` is only available for floats/double. So 2 and 7 are coerced to floats/double, and the result is also a floating point number. So the calculation above is done using floating point arithmetic!

  But in this case, the use of macros was never warranted

  ``` c
  const int THREADS = pow(2,7);
  const int THREADS_PER_BLOCK = 100;
  inline int round_up_div(int a, int b){
      return (1+(a-1)/b);
  }
  const int BLOCKS = round_up_div(THREADS, THREADS_PER_BLOCK);
  ```

  would have provided the same functionality with type checking. While user defined functions might prevent [Constant Folding](https://en.wikipedia.org/wiki/Constant_folding), (in constrast to intinsic functions like `pow`), this  feature could be regained using `pure`/`const` [function attributes](https://stackoverflow.com/questions/2798188/pure-const-function-attributes-in-different-compilers)

  Of course there are certain things like the use of `__LINE__` or `__FILE__` for debugging, which require the use of macros.
