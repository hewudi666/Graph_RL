import matplotlib.pyplot as plt
import numpy as np


def readfile (path1):
    re1 = []
    time1 = []
    re = []
    time = []
    try:
        file = open(path1, 'r')
        # file = open('./output_zhai/uniform-61-n20/0.txt', 'r')
    except FileNotFoundError:
        print('File is not found')
    else:
        lines = file.readlines()
        j = 0
        for line in lines:
            a = line.split()
            re.append(float(a[0]))
            time.append(float(a[1]))
            j += 1
        file.close()
    return re,time
def write(path,M):
    output = open(path, 'w+')
    for i in range(len(M)):
        for j in range(len(M[i])):
            output.write(str(M[i][j]))
            output.write(' ')
        output.write('\n')
    output.close()

Type = ['U_3']
Algorithm = ['TSP-EP','TSP-GP','TSP-MC','HGVNS']
# Num=[20,50,100]
# MA_file=['U_3-61-n20','U_3-71-n50','U_3-92-n100']
Num=[100]
MA_file=['U_3-92-n100']
# Num=[50]
marker=['>','*','s','^']
color=['red','fuchsia','coral','darkslategrey']
Class=[]
i_ma=0
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'
for num in Num:
    curves=[]
    i=0

    ma_path="curve_plot2/MATSP_D/"+str(num)+'/'+MA_file[i_ma]
    i_ma+=1
    Re=np.zeros(2*num+1)
    Time = np.zeros(2*num+1)
    for j in range(30):
        re,time=readfile(ma_path+'-'+str(j)+'.txt')
        print(ma_path+'-'+str(j)+'.txt')
        Re+=re
        Time+=time
    # Time[0]=0.085*30
    plt.plot(Time[::8]/30, Re[::8]/30, label="MATSP-D", linestyle='-',marker="o",c='mediumblue')
    for alg in Algorithm:
            a=0
            path = "curve_plot2/"+alg+"/"+str(num)+".txt"
            print(path)
            x1, y1 = readfile(path)
            curves.append([x1,y1])
            i+=1

    plt.plot(curves[0][1][:20], curves[0][0][:20], label=Algorithm[0], linestyle='-', marker=marker[0], c=color[0])
    plt.plot(curves[1][1][::8], curves[1][0][::8], label=Algorithm[1], linestyle='-', marker=marker[1], c=color[1])
    plt.plot(curves[2][1][::8], curves[2][0][::8], label=Algorithm[2], linestyle='-', marker=marker[2], c=color[2])
    plt.plot(curves[3][1][::8], curves[3][0][::8], label=Algorithm[3], linestyle='-', marker=marker[3], c=color[3])
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    plt.tick_params(labelsize=15)
    plt.xlabel('Runtime/s',fontsize=20)
    plt.ylabel('Total cost',fontsize=20)
      # 将y轴的刻度方向设置向内
    ax = plt.gca()
    bwith = 1.5
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(top='on', bottom='on', left='on', right='on')
    plt.legend(loc='upper right',bbox_to_anchor=(0.95, 0.95),fontsize=30,prop={"family" : "Times New Roman"})

    plt.savefig("cm.png", dpi=600)
    plt.show()



