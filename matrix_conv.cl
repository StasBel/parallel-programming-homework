__kernel void matrix_conv(__global const float *a, __global const float *b, __global float *c,
                          const int n, const int m) {
    int row = get_global_id(0), col = get_global_id(1);

    if (row >= n || col >= n) {
        return;
    }

    float result = 0;
    int nr = 0, nc = 0, hm = m / 2;

    for (int i = -hm; i <= hm; i++) {
        for (int j = -hm; j <= hm; j++) {
            nr = row + i;
            nc = col + j;

            if (nr < 0 || nc < 0 || nr >= n || nc >= n) {
                continue;
            }

            result += a[nr * n + nc] * b[(hm + i) * m + hm + j];
        }
    }

    c[row * n + col] = result;
}