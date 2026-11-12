// N x N output-stationary systolic array, built from pe.v.
// Inputs must already be skewed (see skew.v / systolic_os_top.v): row i of A
// (with its first flag) enters i cycles late, column j of B enters j cycles late.
// Finished results leave at the TOP of each column: col_result[j] with
// col_valid[j], in row order (row 0 first), at most one every other cycle.
// Requires K >= 2N - 1 for back-to-back tiles.
module systolic_os #(
    parameter N    = 4,
    parameter DW   = 8,
    parameter ACCW = 32
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire [N-1:0]         first_left,   // first flag entering row i
    input  wire [N*DW-1:0]      a_left,       // a_left[i*DW +: DW] enters row i
    input  wire [N*DW-1:0]      b_top,        // b_top[j*DW +: DW] enters column j
    output wire [N*ACCW-1:0]    col_result,   // result leaving the top of column j
    output wire [N-1:0]         col_valid     // col_valid[j]: col_result[j] is a real result
);
    // Horizontal wires (A and the first flag): slot i*(N+1)+j enters PE(i,j).
    wire [N*(N+1)*DW-1:0]   a_h;
    wire [N*(N+1)-1:0]      f_h;
    // Vertical wires (B, moving down): slot i*N+j enters PE(i,j) from above.
    wire [(N+1)*N*DW-1:0]   b_v;
    // Result chain (moving up): slot i*N+j is the output of PE(i,j);
    // row N (slots N*N+j) is the constant-zero input below the bottom row.
    wire [(N+1)*N*ACCW-1:0] r_v;
    wire [(N+1)*N-1:0]      v_v;

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_edge
            // Left edge (row i) and top edge (column i) inputs.
            assign a_h[(i*(N+1))*DW +: DW] = a_left[i*DW +: DW];
            assign f_h[i*(N+1)]            = first_left[i];
            assign b_v[i*DW +: DW]         = b_top[i*DW +: DW];
            // Bottom of column i: nothing below.
            assign r_v[(N*N+i)*ACCW +: ACCW] = 0;
            assign v_v[N*N+i]                = 1'b0;
            // Top of column i: results leave here.
            assign col_result[i*ACCW +: ACCW] = r_v[i*ACCW +: ACCW];
            assign col_valid[i]               = v_v[i];
        end

        for (i = 0; i < N; i = i + 1) begin : g_row
            for (j = 0; j < N; j = j + 1) begin : g_col
                pe #(.DW(DW), .ACCW(ACCW)) u_pe (
                    .clk       (clk),
                    .rst       (rst),
                    .first_in  (f_h[i*(N+1)+j]),
                    .a_in      (a_h[(i*(N+1)+j)*DW     +: DW]),
                    .b_in      (b_v[(i*N+j)*DW         +: DW]),
                    .r_in      (r_v[((i+1)*N+j)*ACCW   +: ACCW]),   // from PE(i+1, j)
                    .v_in      (v_v[(i+1)*N+j]),
                    .first_out (f_h[i*(N+1)+j+1]),
                    .a_out     (a_h[(i*(N+1)+j+1)*DW   +: DW]),     // to PE(i, j+1)
                    .b_out     (b_v[((i+1)*N+j)*DW     +: DW]),     // to PE(i+1, j)
                    .r_out     (r_v[(i*N+j)*ACCW       +: ACCW]),   // to PE(i-1, j)
                    .v_out     (v_v[i*N+j])
                );
            end
        end
    endgenerate
endmodule
