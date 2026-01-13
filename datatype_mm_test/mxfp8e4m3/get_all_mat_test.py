import os
import re

bias_type = 1
bias_type_name = ["error", "zeroinit", "rowrepeat", "fullbias"]

def get_all_test():
    for num in range(0, 4):
        print(num)
        with open("cute_Matmul_mxfp8_mnk_64_64_64_zeroinit.c", "r") as f:
            content = f.read()
            #完整的替换一行，以免替换到其他地方
            content = content.replace("#include \"matmul_value_mxfp8_mnk_64_64_64_zeroinit.h\"", f"#include \"matmul_value_mxfp8_mnk_{64 * 2**num}_{64 * 2**num}_{64 * 2**num}_{bias_type_name[bias_type]}.h\"")
            with open(f"cute_Matmul_mxfp8_mnk_{64 * 2**num}_{64 * 2**num}_{64 * 2**num}_{bias_type_name[bias_type]}.c", "w") as f:
                f.write(content)
                    
get_all_test()