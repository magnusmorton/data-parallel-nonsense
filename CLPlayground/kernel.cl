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
        if (gcd(n, i) == 1)
            acc ++;
    }
    return acc;
}

kernel void euler_totient(global int* input, global int* output)
{
    size_t i = get_global_id(0);
    output[i] = phi(input[i]);
}


kernel void parsum(global int* input, global int* partial_sums, local int* localSums)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    
    localSums[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 1; offset < group_size; offset <<= 1) {
        int mask = (offset << 1) - 1;
        if ((local_id & mask) == 0) {
            localSums[local_id] += localSums[offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        partial_sums[get_group_id(0)];
    }

}