import numpy as np
np.set_printoptions(threshold=np.inf)

golden_file = "/home/yuanbin/chipyard/generators/cute/cutetest/transformer/sublayer/QKV_trace.txt"

Q_golden = []
K_golden = []
V_golden = []
QKV_golden = []
attn_golden = []
out_buf_godlen = []

with open(golden_file, "r") as f:
# with open(f"../conv_value_mnk_49_128_128_k1_s1_oh7.h", "r") as f:
    # 读取所有内容
    content = f.read()
    data = content.split("QKV_buf =")[1].split(";")[0]
    QKV_golden = eval(data.replace("{", "[").replace("}", "]"))
    
    data = content.split("attn_buf =")[1].split(";")[0]
    attn_golden = eval(data.replace("{", "[").replace("}", "]"))

    data = content.split("out_buf = ")[1].split(";")[0]
    out_buf_godlen = eval(data.replace("{", "[").replace("}", "]"))

QKV_golden[0] = [QKV_golden[0][j * 512 : (j+1) * 512] for j in range(128)]
QKV_golden[1] = [QKV_golden[1][j * 512 : (j+1) * 512] for j in range(128)]
QKV_golden[2] = [QKV_golden[2][j * 128 : (j+1) * 128] for j in range(512)]
Q_golden = np.array(QKV_golden[0])
K_golden = np.array(QKV_golden[1])
V_golden = np.array(QKV_golden[2])
attn_golden = np.array(attn_golden)
out_buf_godlen = np.array(out_buf_godlen)

def write_matrix(f, matrix, m, n):
    f.write("[\n")
    for i in range(m):
        f.write("[")
        for j in range(n):
            f.write(str(matrix[i][j]) + ",")
        f.write("],\n")
    f.write("];\n")

def write_matrix_3d(f, matrix, num, m, n):
    f.write("[\n")
    for k in range(num):
        f.write("[\n")
        for i in range(m):
            f.write("[")
            for j in range(n):
                f.write(str(matrix[k][i][j]) + ",")
            f.write("],\n")
        f.write("],\n")
    f.write("];\n")

with open(f"../cutetest/transformer-small.h", "r") as f:
    content = f.read()
    data = content.split("static const elem_t Wqkvo[4][512][512] __attribute__((aligned(256))) = ")[1].split(";")[0]
    Wqkvo = eval(data.replace("{", "[").replace("}", "]"))
    # print(martix_A)
    
    # 找到static char b[128][128] __attribute__((aligned(256))) = 后 ; 之前的内容
    data = content.split("static const elem_t input[128][512] __attribute__((aligned(256))) = ")[1].split(";")[0]
    weight = eval(data.replace("{", "[").replace("}", "]"))
    # print(martix_B)

    data = content.split("static const elem_t enc_out[128][512] __attribute__((aligned(256))) =")[1].split(";")[0]
    enc_out = np.array(eval(data.replace("{", "[").replace("}", "]")))

    data = content.split("static const elem_t resadd1_buf[128][512] __attribute__((aligned(256))) =")[1].split(";")[0]
    resadd1_buf = np.array(eval(data.replace("{", "[").replace("}", "]")))

    data = content.split("static const elem_t ff1_w[1024][512] __attribute__((aligned(256))) =")[1].split(";")[0]
    ff1_w = np.array(eval(data.replace("{", "[").replace("}", "]")))

    data = content.split("static const acc_t ff1_b[1024] __attribute__((aligned(256))) =")[1].split(";")[0]
    data = eval(data.replace("{", "[").replace("}", "]"))
    data = [data for i in range(128)]
    ff1_b = np.array(data)
    
    q = np.array(Wqkvo[0])
    k = np.array(Wqkvo[1])
    v = np.array(Wqkvo[2])
    o = np.array(Wqkvo[3])
    input_data = np.array(weight)
    Q_buf = np.dot(input_data, q.T)
    K_buf = np.dot(enc_out, k.T)
    V_buf = np.dot(enc_out, v.T).T
    attn_buf = [np.dot(Q_golden[:,head * 128 : (head+1) * 128], K_golden[:,head * 128 : (head+1) * 128].T) for head in range(4)]
    out_buf = [np.dot(attn_golden[head], V_golden[head * 128 : (head+1) * 128, :].T) for head in range(4)]
    out = np.dot(out_buf_godlen, o.T)
    ff1 = np.dot(resadd1_buf, ff1_w.T) + ff1_b
    with open(f"./Q.txt", "w") as f_write:

        # f_write.write("Q_buf = ")
        # write_matrix(f_write, Q_buf, 128, 512)
        # f_write.write("K_buf = ")
        # write_matrix(f_write, K_buf, 128, 512)
        # f_write.write("V_buf = ")
        # write_matrix(f_write, V_buf, 512, 128)
        # f_write.write("attn_buf = ")
        # write_matrix_3d(f_write, attn_buf, 4, 128, 128)
        # f_write.write("out_buf = ")
        # write_matrix_3d(f_write, out_buf, 4, 128, 128)
        # f_write.write("out = ")
        # write_matrix(f_write, out, 128, 512)
        f_write.write("ff1 = ")
        write_matrix(f_write, ff1, 128, 1024)