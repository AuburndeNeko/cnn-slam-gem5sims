CXXFLAGS = -O3 -std=c++14 -Wall
HDR      = $(wildcard backend/*.hpp)
MAX_LM  ?= 24
XC = docker run --rm -v "$(CURDIR)":/work -w /work ubuntu:20.04 bash -c
XINSTALL = apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++-aarch64-linux-gnu > /dev/null

all: slam slam_mc

# single-core back-end (unchanged)
slam: backend/main.cpp $(HDR)
	$(CXX) $(CXXFLAGS) -DSLAM_MAX_LM=$(MAX_LM) backend/main.cpp -o $@

slam_arm64: backend/main.cpp $(HDR)
	$(XC) "$(XINSTALL) && aarch64-linux-gnu-g++ $(CXXFLAGS) -static -DSLAM_MAX_LM=$(MAX_LM) backend/main.cpp -o $@"

# multi-core pipelined workload (pthreads)
slam_mc: backend/main_mc.cpp $(HDR)
	$(CXX) $(CXXFLAGS) -pthread -DSLAM_MAX_LM=$(MAX_LM) backend/main_mc.cpp -o $@

slam_mc_arm64: backend/main_mc.cpp $(HDR)
	$(XC) "$(XINSTALL) && aarch64-linux-gnu-g++ $(CXXFLAGS) -static -pthread -DSLAM_MAX_LM=$(MAX_LM) backend/main_mc.cpp -o $@"

clean:
	rm -f slam slam_arm64 slam_mc slam_mc_arm64

.PHONY: all clean
