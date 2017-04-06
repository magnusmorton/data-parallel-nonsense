//
//  main.c
//  CLPlayground
//
//  Created by John Morton on 28/01/2016.
//  Copyright © 2016 Magnus. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

#include "kernel.cl.h"

#define RANGE 65536
#define START 4000

int gcd(int n, int k)
{
    int x;
    while (k) {
        x = n;
        n = k;
        k = x % k;
    }
    return abs(n);
}

int phi(int n)
{
    int acc = 0;
    for (int i=0; i< n; i++) {
        if (gcd(n, i) == 1) {
            acc++;
        }
    }
    return acc;
}

void euler_totient(int* input,  int* output)
{
    for (int i = 0; i < RANGE; i++) {
        output[i] = phi(input[i]);
    }
}

int main(int argc, const char * argv[]) {
    static mach_timebase_info_data_t sTimebaseInfo;
    int seq = 0;
    int cpu = 0;
    // insert code here...
    int* data = (int*) malloc(sizeof(cl_int)*RANGE);
    for (int i = 0; i < RANGE; i ++) {
        data[i] = START + i;
        //printf("%d",data[i]);
    }
    
    int* out = (int*) malloc(sizeof(cl_int)*RANGE);
    int* sums = (int*) malloc(sizeof(cl_int)*RANGE);
    printf("Hello, World!\n");
    if (argc > 1){
        if (strncmp(argv[1], "seq", 3) == 0) {
            seq = TRUE;
        } else if (strncmp(argv[1], "cpu", 3) == 0) {
            cpu = 1;
        }
    }
    uint64_t start = mach_absolute_time();
    if (seq)
        euler_totient(data, out);
    else {
        dispatch_queue_t queue;
        if (cpu)
            queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
        else
            queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
       
        void *mem_in = gcl_malloc(sizeof(cl_int)*RANGE, data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        void *mem_out = gcl_malloc(sizeof(cl_int)*RANGE, NULL, CL_MEM_WRITE_ONLY);
        void *mem_sums = gcl_malloc(sizeof(cl_int)*RANGE, NULL, CL_MEM_WRITE_ONLY);
        dispatch_sync(queue, ^{
            size_t wgs = 1;
            if (!cpu){
                gcl_get_kernel_block_workgroup_info(euler_totient_kernel,
                                                   CL_KERNEL_WORK_GROUP_SIZE,
                                                   sizeof(wgs), &wgs, NULL);
            }
            size_t num_wgs;
//            gcl_get_kernel_block_workgroup_info(parsum_kernel, CL_KERNEL_WORK_GROUP_SIZE, <#size_t param_value_size#>, <#void * _Nonnull param_value#>, <#size_t * _Nullable param_value_size_ret#>);
            cl_ndrange range = {
                1,
                {0, 0, 0},
                {RANGE, 0, 0},
                {wgs, 0, 0}
            };
            
            euler_totient_kernel(&range,(cl_int*)mem_in, (cl_int*)mem_out);
            parsum_kernel(&range, (cl_int*)mem_out, (cl_int*)mem_sums, sizeof(cl_int)*wgs);
            
            gcl_memcpy(out, mem_out, sizeof(cl_float) * RANGE);
            gcl_memcpy(sums, mem_sums, sizeof(cl_int) * RANGE);
            
            
        });
        dispatch_release(queue);
    }
    
    // sum everything
    int acc = 0;
    for (int i =0; i < RANGE; i++) {
        acc += out[i];
    }
    uint64_t end = mach_absolute_time();
    if ( sTimebaseInfo.denom == 0 ) {
        mach_timebase_info(&sTimebaseInfo);
    }
    uint64_t elapsed = end - start;
    uint64_t elapsedmillis = (elapsed/1000000) * sTimebaseInfo.numer / sTimebaseInfo.denom;
    printf("SumEuler out: %d\n", acc );
    printf("Time elapsed: %llu milliseconds\n", elapsedmillis);
    return 0;
}
