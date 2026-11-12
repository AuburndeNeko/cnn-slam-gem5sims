`timescale 1ns/1ps
// Testbench: multiplies an N x K matrix A by a K x N matrix B on the array,
// for two different tiles in a row, and checks every C[i][j] against a
// reference computed here. Tile 0 includes the INT8 extremes; tile 1 is random.
module systolic_os_tb;
    localparam N = 4, K = 6, DW = 8, ACCW = 32;

    reg clk = 0, rst = 1, clear = 0;
    reg  [N*DW-1:0]     a_left = 0, b_top = 0;
    wire [N*N*ACCW-1:0] c_flat;

    reg signed [DW-1:0]   A [0:N*K-1];   // A[i][k] stored at A[i*K+k]
    reg signed [DW-1:0]   B [0:K*N-1];   // B[k][j] stored at B[k*N+j]
    reg signed [ACCW-1:0] got;
    integer i, j, k, t, trial, expected, errors = 0, seed = 7;

    systolic_os #(.N(N), .DW(DW), .ACCW(ACCW)) dut (
        .clk(clk), .rst(rst), .clear(clear),
        .a_left(a_left), .b_top(b_top), .c_flat(c_flat)
    );

    always #5 clk = ~clk;

    // Feed one tile with the skew: at cycle t, row i gets A[i][t-i] and
    // column j gets B[t-j][j] (zero outside the matrix).
    task run_tile;
        begin
            for (t = 0; t < K + 2*N - 2; t = t + 1) begin
                for (i = 0; i < N; i = i + 1) begin
                    k = t - i;
                    a_left[i*DW +: DW] = (k >= 0 && k < K) ? A[i*K+k] : 0;
                end
                for (j = 0; j < N; j = j + 1) begin
                    k = t - j;
                    b_top[j*DW +: DW] = (k >= 0 && k < K) ? B[k*N+j] : 0;
                end
                @(posedge clk); #1;
            end
            a_left = 0; b_top = 0;
            @(posedge clk); #1;    // one extra edge: the PE is pipelined (latency 1)
        end
    endtask

    // Compare every output with a reference matrix multiply.
    task check_tile;
        begin
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    expected = 0;
                    for (k = 0; k < K; k = k + 1)
                        expected = expected + A[i*K+k] * B[k*N+j];
                    got = c_flat[(i*N+j)*ACCW +: ACCW];
                    if (got !== expected) begin
                        $display("FAIL tile %0d C[%0d][%0d]: got %0d expected %0d",
                                 trial, i, j, got, expected);
                        errors = errors + 1;
                    end
                end
        end
    endtask

    initial begin
        @(posedge clk); #1 rst = 0;

        for (trial = 0; trial < 2; trial = trial + 1) begin
            for (k = 0; k < N*K; k = k + 1) begin
                A[k] = $random(seed) % 128;
                B[k] = $random(seed) % 128;
            end
            if (trial == 0) begin            // signed extremes
                A[0] = -128; B[0] = -128;
                A[1] = 127;  B[N] = 127;
            end

            clear = 1; @(posedge clk); #1 clear = 0;   // start a new tile
            run_tile;
            check_tile;
        end

        if (errors == 0) $display("PASS: %0dx%0d array, K=%0d, 2 tiles, all outputs correct", N, N, K);
        else             $display("%0d error(s)", errors);
        $finish;
    end
endmodule
