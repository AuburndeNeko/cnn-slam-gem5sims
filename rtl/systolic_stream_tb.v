`timescale 1ns/1ps
// Streaming + readout testbench: three tiles fed back to back with NO gap,
// then a flush. Results are read ONLY from the top edge of the array.
// Each column must deliver its results in order: tile 0 rows 0..N-1, then
// tile 1, then tile 2, each matching a reference matrix multiply.
module systolic_stream_tb;
    localparam N = 4, K = 8, T = 3, DW = 8, ACCW = 32;   // K >= 2N - 1 required

    reg clk = 0, rst = 1, first = 0;
    reg  [N*DW-1:0]   a_col = 0, b_row = 0;
    wire [N*ACCW-1:0] col_result;
    wire [N-1:0]      col_valid;

    reg signed [DW-1:0] A   [0:T*N*K-1];   // tile tt: A[i][k] at (tt*N+i)*K+k
    reg signed [DW-1:0] B   [0:T*K*N-1];   // tile tt: B[k][j] at (tt*K+k)*N+j
    integer             EXP [0:T*N*N-1];   // tile tt: C[i][j] at tt*N*N+i*N+j
    integer             cnt [0:N-1];       // results received so far, per column
    reg signed [ACCW-1:0] got;
    integer i, j, k, t, tt, p, m, mt, mi, errors = 0, seed = 5;

    systolic_os_top #(.N(N), .DW(DW), .ACCW(ACCW)) dut (
        .clk(clk), .rst(rst), .first(first),
        .a_col(a_col), .b_row(b_row),
        .col_result(col_result), .col_valid(col_valid)
    );

    always #5 clk = ~clk;

    // Monitor the top edge: the n-th result out of column m must be
    // tile n / N, row n % N, column m.
    always @(posedge clk) begin
        #1;
        for (m = 0; m < N; m = m + 1)
            if (!rst && col_valid[m]) begin
                if (cnt[m] >= T*N) begin
                    $display("FAIL column %0d: more results than expected", m);
                    errors = errors + 1;
                end else begin
                    mt  = cnt[m] / N;
                    mi  = cnt[m] % N;
                    got = col_result[m*ACCW +: ACCW];
                    if (got !== EXP[mt*N*N + mi*N + m]) begin
                        $display("FAIL tile %0d C[%0d][%0d]: got %0d expected %0d",
                                 mt, mi, m, got, EXP[mt*N*N + mi*N + m]);
                        errors = errors + 1;
                    end
                end
                cnt[m] = cnt[m] + 1;
            end
    end

    initial begin
        if (K < 2*N - 1) $display("WARNING: K=%0d < 2N-1=%0d, back-to-back tiles will collide", K, 2*N-1);
        for (p = 0; p < N; p = p + 1) cnt[p] = 0;

        // Random tiles; tile 0 also gets the INT8 extremes.
        for (p = 0; p < T*N*K; p = p + 1) begin
            A[p] = $random(seed) % 128;
            B[p] = $random(seed) % 128;
        end
        A[0] = -128; B[0] = -128; A[1] = 127; B[N] = 127;

        // Reference results.
        for (tt = 0; tt < T; tt = tt + 1)
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    EXP[tt*N*N + i*N + j] = 0;
                    for (k = 0; k < K; k = k + 1)
                        EXP[tt*N*N + i*N + j] = EXP[tt*N*N + i*N + j]
                            + A[(tt*N+i)*K+k] * B[(tt*K+k)*N+j];
                end

        @(posedge clk); #1 rst = 0;

        // Stream all tiles back to back.
        for (tt = 0; tt < T; tt = tt + 1)
            for (t = 0; t < K; t = t + 1) begin
                first = (t == 0);
                for (i = 0; i < N; i = i + 1)
                    a_col[i*DW +: DW] = A[(tt*N+i)*K+t];
                for (j = 0; j < N; j = j + 1)
                    b_row[j*DW +: DW] = B[(tt*K+t)*N+j];
                @(posedge clk); #1;
            end

        // Flush: a first flag with zero data hands over the last tile.
        first = 1; a_col = 0; b_row = 0;
        @(posedge clk); #1 first = 0;
        repeat (4*N + 4) @(posedge clk);
        #2;

        for (p = 0; p < N; p = p + 1)
            if (cnt[p] !== T*N) begin
                $display("FAIL column %0d: received %0d results, expected %0d", p, cnt[p], T*N);
                errors = errors + 1;
            end

        if (errors == 0) $display("PASS: %0dx%0d array, %0d tiles streamed, K=%0d, all %0d results read out in order",
                                  N, N, T, K, T*N*N);
        else             $display("%0d error(s)", errors);
        $finish;
    end
endmodule
