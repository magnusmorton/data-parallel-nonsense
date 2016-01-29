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