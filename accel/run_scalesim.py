import sys
from scalesim.scale_sim import scalesim

cfg, topo, out = sys.argv[1:4]
gemm = (sys.argv[4] if len(sys.argv) > 4 else 'gemm') == 'gemm'
s = scalesim(save_disk_space=True, verbose=True, config=cfg, topology=topo,
             layout="./layouts/conv_nets/test.csv", input_type_gemm=gemm)
s.run_scale(top_path=out)