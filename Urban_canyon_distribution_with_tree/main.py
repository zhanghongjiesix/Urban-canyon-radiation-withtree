import numpy as np
import pandas as pd
from decimal import Decimal
from viewfactor import *
from shade import *

class Radiation_tree:
    def __init__(self, canyon_w: float, canyon_h: float, canyon_azimuth: float, w_layer: list, left_layer: list, right_layer: list,
                 all_albedo_layer: list,  azimuth_s: float, zenith_s: float,
                 dir_solar: float, dif_solar: float, tree=(((1, 1), 0, 0, 0),)):
        self.w = canyon_w   # 路面宽度
        self.h = canyon_h   # 街道峡谷高度
        self.w_layer = np.array(w_layer)  # 街道峡谷路面分层，并转化为ndarray对象
        self.left_layer = np.array(left_layer)  # 街道峡谷左墙面的分层，并转化为ndarray对象
        self.right_layer = np.array(right_layer)  # 街道峡谷右墙面分层，并转化为ndarray对象
        self.albedo_layer = np.array(all_albedo_layer)  # 街道峡谷所有表面的反射率，左上开始赋值到右上，并转化为ndarray对象
        self.azimuth_c = (canyon_azimuth * np.pi) / 180  # 街道峡谷的方位角并转化为弧度值
        self.azimuth_s = (azimuth_s * np.pi) / 180  # 太阳方位角并转化为弧度值
        self.zenith_s = (zenith_s * np.pi) / 180  # 太阳天顶角并转化为弧度值
        self.dir_solar = dir_solar  # 太阳直射辐射量
        self.dif_solar = dif_solar  # 太阳散射辐射量
        self.tree = tree
        # self.tree = kwargs.get('tree', ((1, 1), 0, 0, 0))  # 树木的圆心,半径,叶面积密度和种植密度，可以根据需要调整树木的个数
        assert Decimal(self.left_layer.sum()).quantize(Decimal('0.00')) == Decimal(self.h).quantize(Decimal('0.00')), '左侧建筑总高度与分层高度不一致'
        assert Decimal(self.right_layer.sum()).quantize(Decimal('0.00')) == Decimal(self.h).quantize(Decimal('0.00')),  '右侧建筑总高度与分层高度不一致'
        assert Decimal(self.w_layer.sum()).quantize(Decimal('0.00')) == Decimal(self.w).quantize(Decimal('0.00')), '路面宽度与路面分层不一致'
        assert np.size(self.left_layer) + np.size(self.right_layer) + np.size(self.w_layer) == np.size(self.albedo_layer), '反射率个数与分层个数不一致'
        assert 0 <= self.azimuth_c < np.pi, '街道峡谷方位角应该在0-pi之间'
        assert 0 <= self.zenith_s < 0.5 * np.pi, '太阳天顶角应该在0-0.5pi之间'
        assert 0 <= self.azimuth_s <= 2 * np.pi, '太阳方位角应在0-2pi之间'
        assert self.azimuth_c != self.azimuth_s, '请避免街道方位角与太阳方位角相同'
        assert all(0 < self.albedo_layer[x] < 1 for x in np.arange(0, self.albedo_layer.size)), '表面反射率应该在0-1之间'
        for i in np.arange(0, len(self.tree)):
            assert self.tree[i][0][1] + self.tree[i][1] <= self.h, '请避免树冠顶高于街道峡谷'

    def geo(self):
        '''
        Returns 街道峡谷表面的坐标值，左上开始到右上，逆时针转动
        -------
        '''
        geo_list = []
        # 建立从左上开始的各个面的起点和重点坐标，生成一个ndarray[[x1,y1],[x2,y2],[x3,y3],.......]
        # 对于左面墙，X==0，所以只需要赋值Y即可
        for i in np.arange(0, self.left_layer.size):
            y = self.h - self.left_layer[0:i].sum()
            geo_list.append([0, y])
        # 对于路面，Y==0，所以只需要赋值x即可
        for i in np.arange(0, self.w_layer.size):
            x = self.w_layer[0:i].sum()
            geo_list.append([x, 0.0])
        # 对于右墙面，x==self.w，所以只需要赋值y即可
        for i in np.arange(0, self.right_layer.size + 1):
            y = self.right_layer[0:i].sum()
            geo_list.append([self.w, y])
        geo = np.array(geo_list)
        return geo

    def view_mat(self):         # 这个算出来的面A对面A的角系数为-1，后面需要变一下
        '''
        Returns 街道峡谷内表面之间的角系数矩阵；天空视角因子矩阵；天空对树冠的角系数矩阵
        -------
        '''
        wall_view_mat = []
        sky_view_mat = []
        tree_view_mat = []
        geo = self.geo()
        for i in range(0, len(geo) - 1):
            for j in range(0, len(geo) - 1):
                slice_a = [geo[i], geo[i + 1]]
                slice_b = [geo[j], geo[j + 1]]
                view = lineviewfactor(slice_a, slice_b)
                wall_view_mat.append(view)
        wall_view_mat = np.array(wall_view_mat).reshape(len(geo) - 1, len(geo) - 1).T
        for i in range(0, len(geo) - 1):
            slice_a = [geo[i], geo[i + 1]]
            sky = [geo[len(geo) - 1], geo[0]]
            view = lineviewfactor(slice_a, sky)
            sky_view_mat.append(view)
        sky_view_mat = np.array(sky_view_mat)
        for i in range(0, len(self.tree)):
            tree = [self.tree[i][0], self.tree[i][1]]
            sky = [geo[len(geo) - 1], geo[0]]
            view = treeviewfactor(sky, tree)
            tree_view_mat.append(view)
        tree_view_mat = np.array(tree_view_mat)
        return np.around(wall_view_mat, 2), np.around(sky_view_mat, 2), np.around(tree_view_mat, 2)

    def abs_tree(self):
        '''
        Returns  返回树冠对初始太阳散射辐射的吸收率，峡谷中所有的树冠在多重反射过程中对太阳辐射的吸收率, 单颗树木的吸收率矩阵
        -------
        '''
        '''求树冠对初始太阳散射辐射的吸收率'''
        trans = []                                          # 树木1到树木i的透射率列表
        pro = []                                            # 光线与树木的相交概率列表
        for i in range(0, len(self.tree)):
            '''文献中的系数为0.61，我们计算的系数为0.328'''
            trans_tree = np.exp(-0.61 * self.tree[i][2])   # 根据单株树木透射率计算公式计算透射率。
            trans.append(trans_tree)
        trans_tree_mat = np.array(trans)                    # 树冠对太阳辐射的透射率矩阵
        abs_tree_mat = 1 - trans_tree_mat                   # 树冠对太阳辐射的吸收率矩阵
        tree_view_mat = self.view_mat()[2]                  # 天空对树冠的角系数
        abs_ini_dif = (abs_tree_mat * tree_view_mat).sum()  # 所有的树冠对初始太阳散射辐射的吸收率
        '''求在整个多重反射过程中的吸收率'''
        for i in range(0, len(self.tree)):
            pro_i = np.pi * self.tree[i][1] ** 2 * self.tree[i][3] / (self.w * self.h)
            pro.append(pro_i)
        pro_tree_mat = np.array(pro)                        # 光线通过树冠的概率矩阵
        abs_solar = (pro_tree_mat * abs_tree_mat).sum()     # 峡谷中所有的树冠对太阳辐射的吸收率at
        return abs_ini_dif, abs_solar, abs_tree_mat

    def dif_ini(self):
        '''
        Returns 街道峡谷表面初始散射太阳辐射量矩阵
        -------
        '''
        abs_tree = self.abs_tree()[0]
        sky_view = self.view_mat()[1]
        # 先求被树木遮挡后的太阳散射辐射量
        solar_dif = self.dif_solar * (1 - abs_tree)
        # 根据天空视角因子求每个面的太阳散射辐射量
        dif_ini = solar_dif * sky_view
        return dif_ini

    def dir_ini(self):
        '''
        Returns 街道峡谷表面初始直射太阳辐射量矩阵
        -------
        '''
        geo = self.geo()
        sky = [geo[len(geo) - 1], geo[0]]
        tree_abs_mat = np.array([0] * (len(geo) - 1)).astype(float)
        for i in np.arange(0, len(self.tree)):
            tree = self.tree
            tree_cordinate = shadecoordinate(sky=sky, tree=[tree[i][0], tree[i][1]], azimuth_canyon=self.azimuth_c, azimuth_solar=self.azimuth_s, zenith_solar=self.zenith_s)
            # 求树冠造成的遮阴矩阵
            tree_shade_mat = shade_mat(wall=geo.tolist(), shadecor=tree_cordinate[1])
            tree_abs_mat += (tree_shade_mat * tree[i][-1] * self.abs_tree()[-1][i])        # 所有树对太阳辐射的吸收矩阵
        # 求墙的吸收率矩阵
        building_cor = shadecoordinate(sky=sky, azimuth_canyon=self.azimuth_c, azimuth_solar=self.azimuth_s, zenith_solar=self.zenith_s)
        building_abs_mat = shade_mat(wall=geo.tolist(), shadecor=building_cor[0])        # 墙面的吸收率矩阵
        allshade_abs_mat = tree_abs_mat + building_abs_mat                               # 墙面和树冠对太阳直射辐射的吸收率矩阵
        allreceive_abs_mat = 1 - allshade_abs_mat
        allreceive_abs_mat = np.where(allreceive_abs_mat < 0, 0, allreceive_abs_mat)      # 墙面接收太阳辐射的矩阵，当为1时表示没有任何遮挡，0表示全部遮挡
        # print(allreceive_abs_mat)
        '''接下来求假如没有遮挡的情况下每个面接收到的太阳辐射矩阵'''
        dir_wall = self.dir_solar * np.tan(self.zenith_s) * np.abs(np.sin(self.azimuth_s - self.azimuth_c))
        dir_ground = self.dir_solar
        dir_all = [dir_wall] * len(self.left_layer) + [dir_ground] * len(self.w_layer) + [dir_wall] * len(self.right_layer)
        # print(dir_all)
        # 求有树木和建筑遮挡造成的初始直射辐射矩阵
        dir_ini = allreceive_abs_mat * dir_all
        # print(dir_ini)
        return dir_ini

    def input_para(self):
        '''
        Returns 解方程组需要的参数矩阵
        -------
        '''
        F_mat = self.view_mat()[0]                          # 角系数矩阵 对应公式中的F
        all_albedo = self.albedo_layer
        abs_tree = self.abs_tree()[1]
        all_albedo_mat = []
        for i in np.arange(0, len(all_albedo)):
            i = all_albedo
            all_albedo_mat.append(i)
        all_albedo_mat = np.array(all_albedo_mat).T         # 反射率矩阵
        left_mat = F_mat * all_albedo_mat * (abs_tree - 1)
        row, col = np.diag_indices_from(left_mat)
        left_mat[row, col] = 1
        # 接下来求解初始太阳辐射量,用来计算有效反射率：初始直射加散射
        dif_ini = self.dif_ini()
        dir_ini = self.dir_ini()
        solar_ini = np.array(dif_ini + dir_ini)
        right_mat = (solar_ini * np.array(all_albedo))      # 方程右侧的系数矩阵

        return left_mat, right_mat, solar_ini

    def energy_balance(self):
        '''
        Returns 无限次多重反射后各个表面反射的太阳辐射量
        -------
        '''
        left_mat, right_mat = self.input_para()[0], self.input_para()[1]
        # 计算面从1开始，无限次反射后反射的太阳辐射量
        reflected_radiation = np.linalg.solve(left_mat, right_mat)
        return reflected_radiation

    def solar_potential(self):
        '''
        Returns 各表面接收到的太阳辐射量
        -------
        '''
        solar_reflected = self.energy_balance()
        all_albedo = self.albedo_layer
        solar_potential = solar_reflected / all_albedo
        return solar_potential

    # def solar_potential_new(self):
    #     '''
    #     Returns 考虑多重反射后各个面接收到的太阳辐射量
    #     -------
    #     '''


    def effective_albedo(self):
        '''
        Returns 假设没有树木存在时到达街道峡谷表面的太阳辐射量为入射到街道峡谷的总太阳辐射量，即分母
        ------- 分子为：1 有树木的情况下墙面吸收的太阳辐射量 2 无限次反射过程中树冠吸收的太阳辐射量为：加入没有树时各个面反射的辐射量 - 有树存在时各个面反射的辐射量
        '''
        geo = self.geo()
        sky = [geo[len(geo) - 1], geo[0]]
        dir_solar = self.dir_solar
        dif_solar = self.dif_solar
        sky_view = self.view_mat()[1]
        dif_ini = dif_solar * sky_view                     # 没有树木存在情况下初始散射辐射量
        building_cor = shadecoordinate(sky=sky, azimuth_canyon=self.azimuth_c, azimuth_solar=self.azimuth_s,
                                       zenith_solar=self.zenith_s)
        building_abs_mat = shade_mat(wall=geo.tolist(), shadecor=building_cor[0])  # 墙面的吸收率矩阵
        '''求假如没有树木遮挡的情况下每个面接收到的太阳辐射矩阵'''
        dir_wall = self.dir_solar * np.tan(self.zenith_s) * np.abs(np.sin(self.azimuth_s - self.azimuth_c))
        dir_ground = self.dir_solar
        dir_all = [dir_wall] * len(self.left_layer) + [dir_ground] * len(self.w_layer) + [dir_wall] * len(
            self.right_layer)
        dir_ini = dir_all * (1 - building_abs_mat)
        # step1: 假如没有树木存在时到达街道峡谷的太阳辐射量即分母
        solar_ini = (dir_ini + dif_ini).sum()                     # 分母
        print('进入街道峡谷的太阳辐射能为{}'.format(solar_ini))
        '''求有树木时的初始太阳辐射能量'''
        solar_ini_tree = self.dif_ini() + self.dir_ini()
        print('有树木存在时街道峡谷表面初始太阳辐射能为{}'.format(solar_ini_tree))
        trans_tree_mat = 1 - self.abs_tree()[-1]
        print('树冠对太阳辐射的透射率矩阵为{}'.format(trans_tree_mat))
        solar_potential_tree = self.solar_potential().sum()
        print('有树木存在时街道峡谷表面达到的总太阳辐射能为{}'.format(solar_potential_tree))
        # step2：有树木存在情况下多重反射过程中墙面吸收的太阳辐射量
        all_abs = 1 - self.albedo_layer          # 所有面吸射率ndarray
        wall_abs = (self.solar_potential() * all_abs).sum()        # 所有强面吸收的太阳辐射量 分子1
        print('墙面吸收的辐射量为{}'.format(wall_abs))
        # step3：多重反射过程中树木吸收的太阳辐射量
        all_reflected = self.energy_balance().sum()
        abs_tree = self.abs_tree()[1]                              # 多重反射过程中树木的吸收率
        print('多重反射过程中树木的吸收率为{}'.format(abs_tree))
        tree_abs = all_reflected / (1 - abs_tree) - all_reflected  # 多重反射过程中树木吸收的吸收量
        print('树木吸收的辐射量为{}'.format(tree_abs))
        # tree_abs = all_reflected * abs_tree
        # stpe4：求有效反射率
        effective_albedo = 1 - (wall_abs + tree_abs) / solar_ini
        return effective_albedo

if __name__ == '__main__':
    raidation_tree = Radiation_tree(
        canyon_w=2.0, canyon_h=2.1, canyon_azimuth=0,
        left_layer=[0.42, 0.42, 0.42, 0.42, 0.42],
        w_layer=[0.4, 0.4, 0.4, 0.4, 0.4],
        right_layer=[0.42, 0.42, 0.42, 0.42, 0.42],
        all_albedo_layer=[0.8, 0.8, 0.8, 0.8, 0.8,
                          0.1, 0.1, 0.1, 0.1, 0.1,
                          0.8, 0.8, 0.8, 0.8, 0.8],
        azimuth_s=283, zenith_s=76, dir_solar=33, dif_solar=7,
        # tree=(((1, 1.15), 0.75, 2.49, 1),),
    )
# print(raidation_tree.geo(), type(raidation_tree.geo()))
# print(raidation_tree.view_mat())
# print(raidation_tree.abs_tree())
# print(raidation_tree.dif_ini())
# print(raidation_tree.dir_ini())
# raidation_tree.input_para()
# print(raidation_tree.energy_balance())
# print(raidation_tree.solar_potential().sum())
# print(raidation_tree.effective_albedo())
print(raidation_tree.solar_potential())



