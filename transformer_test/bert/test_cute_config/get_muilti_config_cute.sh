mkdir build
mkdir log
mkdir tmp
cd ../../../../../sims/verilator/get_muilti_config_cute.sh

make CONFIG=CUTE4TopsShuttle512D512V512M512Sysbus512Membus1CoreConfig -j24
