mkdir build
mkdir log
mkdir tmp
cd ../../../../../sims/verilator/get_muilti_config_cute.sh
make CONFIG=CUTE1TopsConfig -j24
make CONFIG=CUTE2TopsConfig -j24
make CONFIG=CUTE4TopsConfig -j24
make CONFIG=CUTE8TopsConfig -j24
make CONFIG=CUTE16TopsConfig -j24
make CONFIG=CUTE32TopsConfig -j24
make CONFIG=CUTE64TopsConfig -j24
cp ./*TopsConfig ../../generators/cute/cutetest/gemm_test/multi_cute_config_test/build/
./test_all_per_K.py