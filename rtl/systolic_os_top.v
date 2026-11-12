// Systolic array with built-in input skew, tile streaming and column readout.
// Each cycle the caller supplies one column of A (a_col[i] = A[i][k]) and the
// matching row of B (b_row[j] = B[k][j]). `first` is high in the cycle that
// carries k = 0 of a new tile. Tiles can follow each other with no gap, as
// long as K >= 2N - 1. After the last tile, one cycle with first = 1 (data
// zero) flushes it out. Results leave on col_result / col_valid.
module systolic_os_top #(
    parameter N    = 4,
    parameter DW   = 8,
    parameter ACCW = 32
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              first,       // high with column/row 0 of a new tile
    input  wire [N*DW-1:0]   a_col,       // a_col[i*DW +: DW] = A[i][k]
    input  wire [N*DW-1:0]   b_row,       // b_row[j*DW +: DW] = B[k][j]
    output wire [N*ACCW-1:0] col_result,  // result leaving the top of column j
    output wire [N-1:0]      col_valid
);
    // The first flag rides in the A skew stage as a 9th bit of each lane,
    // so it is delayed exactly like the A values it belongs to.
    wire [N*(DW+1)-1:0] a_ext, a_ext_skewed;
    wire [N*DW-1:0]     a_skewed, b_skewed;
    wire [N-1:0]        first_skewed;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_pack
            assign a_ext[i*(DW+1) +: DW+1]  = {first, a_col[i*DW +: DW]};
            assign a_skewed[i*DW +: DW]     = a_ext_skewed[i*(DW+1) +: DW];
            assign first_skewed[i]          = a_ext_skewed[i*(DW+1) + DW];
        end
    endgenerate

    skew #(.N(N), .DW(DW+1)) u_skew_a (.clk(clk), .rst(rst), .in_vec(a_ext), .out_vec(a_ext_skewed));
    skew #(.N(N), .DW(DW))   u_skew_b (.clk(clk), .rst(rst), .in_vec(b_row), .out_vec(b_skewed));

    systolic_os #(.N(N), .DW(DW), .ACCW(ACCW)) u_array (
        .clk(clk), .rst(rst),
        .first_left(first_skewed), .a_left(a_skewed), .b_top(b_skewed),
        .col_result(col_result), .col_valid(col_valid)
    );
endmodule
