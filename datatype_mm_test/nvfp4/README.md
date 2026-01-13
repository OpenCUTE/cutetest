编译.h文件的生成程序，里面有随机生成的测试数据，同时包含了对应的golden结果

该文件依赖`myrandom.h`和`fmac.h`、`FloatDecode.h`这三个文件是`cute-fpe/ccode/`里的文件的软链接
```
g++ get_matrix_test.c -o get_matrix_test
```

`get_mattest_value.py`调用`get_matrix_test`生成不同规模的.h文件
```
python3 get_mattest_value.py
```

`get_all_mat_test.py`生成.c测试文件
```
python3 get_all_mat_test.py
```

然后make编译测试文件

`compare_result.py`可以将运行生成.out文件中的trace与.h中的golden进行对比