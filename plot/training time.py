import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

# Graph setting
# 刻度在内，设置刻度字体大小
plt.figure(figsize=[5.0, 5.0])
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.tick_params(labelsize=15, width=1)
plt.xticks(size=15)

algorithm = ['MADDPG', 'PPO', 'GRL'] # 对比算法名称
time_cost = [104.8, 19.1, 3.76]
color = ['lightsalmon','lightgreen','lightskyblue']

plt.bar(algorithm, time_cost, color = color,width=0.4)
plt.ylabel('Training Time Cost  [hour]', fontsize=15)

plt.ylim(0, 120)

for a, b, i in zip(algorithm, time_cost, range(len(algorithm))):
    plt.text(a,b+1, time_cost[i],ha='center',fontsize=20)

plt.savefig("training time cost.png", dpi=600)
plt.show()


class setfig():
    '''
       在绘图前对字体类型、字体大小、分辨率、线宽、输出格式进行设置.
       para colume = 1.半栏图片 7*6cm
                     2.双栏长图 14*6cm
       x轴刻度默认为整数
       手动保存时，默认输出格式为 pdf
       案例 Sample.1:
            fig=setfig(column=2)
            plt.semilogy(x, color='blue', linestyle='solid', label='信号1')
            plt.legend(loc='upper left')
            plt.xlabel('时间/t')
            plt.ylabel('幅度')
            plt.title('冲击声信号')
            fig.show()
    '''

    def __init__(self, column):

        self.column = column  # 设置栏数
        # 对尺寸和 dpi参数进行调整
        plt.rcParams['figure.dpi'] = 400

        # 字体调整
        plt.rcParams['font.sans-serif'] = ['simhei']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
        plt.rcParams['font.weight'] = 'light'
        plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
        plt.rcParams['axes.titlesize'] = 8  # 标题字体大小
        plt.rcParams['axes.labelsize'] = 7  # 坐标轴标签字体大小
        plt.rcParams['xtick.labelsize'] = 7  # x轴刻度字体大小
        plt.rcParams['ytick.labelsize'] = 7  # y轴刻度字体大小
        plt.rcParams['legend.fontsize'] = 6

        # 线条调整
        plt.rcParams['axes.linewidth'] = 1

        # 刻度在内，设置刻度字体大小
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        # 设置输出格式为PDF
        plt.rcParams['savefig.format'] = 'pdf'
        plt.rcParams['figure.autolayout'] = True

    @property
    def tickfont(self):
        plt.tight_layout()
        ax1 = plt.gca()  # 获取当前图像的坐标轴
        # 更改坐标轴字体，避免出现指数为负的情况
        tick_font = font_manager.FontProperties(family='it', size=7.0)
        ax1.xaxis.set_major_locator
        for labelx in ax1.get_xticklabels():
            labelx.set_fontproperties(tick_font)
        for labely in ax1.get_yticklabels():
            labely.set_fontproperties(tick_font)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数

    @property
    def Global_font(self):
        # 设置基本字体
        plt.rcParams['font.sans-serif'] = ['simhei']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
        plt.rcParams['font.weight'] = 'light'

    def show(self):
        # 改变字体
        self.Global_font
        self.tickfont
        # 改变图像大小
        cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
        gcf = plt.gcf()  # 获取当前图像
        if self.column == 1:
            gcf.set_size_inches(7 * cm_to_inc, 6 * cm_to_inc)
        else:
            gcf.set_size_inches(14 * cm_to_inc, 6 * cm_to_inc)

        plt.show()


