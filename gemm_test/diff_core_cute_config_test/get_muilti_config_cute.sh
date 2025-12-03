mkdir build
mkdir log
mkdir tmp
cd ../../../../../sims/verilator
make CONFIG=CUTE2TopsSmallRocketConfig -j24
make CONFIG=CUTE2TopsSmallBoomConfig -j24
cp ./*CUTE2TopsSmallRocketConfig ./*CUTE2TopsSmallBoomConfig ../../generators/cute/cutetest/gemm_test/diff_core_cute_config_test/build/
./test_all_per_K.py