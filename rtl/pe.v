// Output-stationary systolic array processing element (PE), pipelined.
// Stage 1 multiplies, stage 2 accumulates. A 1-bit "first" flag travels with
// the A values; when it arrives, the PE puts its finished sum into the result
// chain and starts the next tile, so tiles stream back to back.
// Result chain: finished results move UP the column one PE per cycle
// (r_in/v_in from the PE below, r_out/v_out to the PE above) and leave at the
// top edge. This needs K >= 2N - 1 so a column drains before its next tile ends.
module pe #(
    parameter DW   = 8,    // input width (INT8)
    parameter ACCW = 32    // accumulator width
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    first_in,   // high with the first A value of a new tile
    input  wire signed [DW-1:0]    a_in,       // from the PE on the left
    input  wire signed [DW-1:0]    b_in,       // from the PE above
    input  wire signed [ACCW-1:0]  r_in,       // result chain, from the PE below
    input  wire                    v_in,       // valid bit for r_in
    output reg                     first_out,  // to the PE on the right (travels with A)
    output reg  signed [DW-1:0]    a_out,      // to the PE on the right
    output reg  signed [DW-1:0]    b_out,      // to the PE below
    output reg  signed [ACCW-1:0]  r_out,      // result chain, to the PE above
    output reg                     v_out       // valid bit for r_out
);
    reg signed [ACCW-1:0] acc;       // running sum for the current tile
    reg signed [2*DW-1:0] prod_r;    // stage-1 register: the product
    reg                   first_r;   // first flag, lined up with prod_r
    reg                   started;   // 1 once a tile has begun (no result before that)

    always @(posedge clk) begin
        if (rst) begin
            first_out <= 0;
            a_out     <= 0;
            b_out     <= 0;
            prod_r    <= 0;
            first_r   <= 0;
            acc       <= 0;
            started   <= 0;
            r_out     <= 0;
            v_out     <= 0;
        end else begin
            // Stage 1: multiply, and pass A, B and the flag on.
            first_out <= first_in;
            a_out     <= a_in;
            b_out     <= b_in;
            prod_r    <= a_in * b_in;     // signed 8x8 -> signed 16-bit product
            first_r   <= first_in;

            // Stage 2.
            if (first_r) begin
                r_out   <= acc;           // previous tile done: put it on the chain
                v_out   <= started;       // (nothing to report before the first tile)
                started <= 1;
                acc     <= prod_r;        // start the new tile with this product
            end else begin
                r_out   <= r_in;          // pass results from below upward
                v_out   <= v_in;
                acc     <= acc + prod_r;  // signed + signed: prod_r is sign-extended
            end
        end
    end
endmodule
