mkdir build
mkdir log
mkdir tmp
cd ../../../../../sims/verilator/get_muilti_config_cute.sh
make CONFIG=CUTE05TopsSCP64Config -j24
make CONFIG=CUTE1TopsSCP128Config -j24
make CONFIG=CUTE2TopsSCP256Config -j24
make CONFIG=CUTE4TopsSCP512Config -j24

make CONFIG=CUTE1TopsSCP64Config -j24
make CONFIG=CUTE2TopsSCP128Config -j24
make CONFIG=CUTE4TopsSCP256Config -j24
make CONFIG=CUTE8TopsSCP512Config -j24

make CONFIG=CUTE2TopsSCP64Config -j24
make CONFIG=CUTE4TopsSCP128Config -j24
make CONFIG=CUTE8TopsSCP256Config -j24
make CONFIG=CUTE16TopsSCP512Config -j24

make CONFIG=CUTE4TopsSCP64Config -j24
make CONFIG=CUTE8TopsSCP128Config -j24
make CONFIG=CUTE16TopsSCP256Config -j24
make CONFIG=CUTE32TopsSCP512Config -j24
cp ./*TopsConfig ../../generators/cute/cutetest/gemm_test/multi_cute_config_test/build/
./test_all_per_K.py