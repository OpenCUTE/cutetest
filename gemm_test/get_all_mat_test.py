import os
import re

def get_all_test():
    for num in range(17, 65):
        print(num)
        with open("cute_Matmul_mnk_256_256_256_zeroinit_transpose.c", "r") as f:
            content = f.read()
            #完整的替换一行，以免替换到其他地方
            content = content.replace("#include \"matmul_value_mnk_256_256_256_zeroinit_transpose.h\"", f"#include \"matmul_value_mnk_512_512_{num*256}_zeroinit_transpose.h\"")
            with open(f"cute_Matmul_mnk_512_512_{num*256}_zeroinit_transpose.c", "w") as f:
                f.write(content)
                    
get_all_test()