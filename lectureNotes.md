# Command line

- `lscpu` information about `native` CPU flags
- `top` (`htop`) provides (detailed) information about running processes
- `kill -9 pid` where `pid` is the process identification, kills this process
- `ssh username@hostIPAdress` log into a remote server
- `scp fromFileLocal.txt username@hostIPAdress:toFile.txt` copy over ssh

# Compiler

## Precompiler

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

## General Advice

TODO:

Local variables -> caches/registries

array of structs vs struct of arrays

precision of sums

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

make sure that your computer can use these types and operations (cf. [Command line](#Command Line), Compiler Constants, Architecture flags)

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

