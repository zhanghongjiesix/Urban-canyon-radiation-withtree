import numpy as np
import math


def shadecoordinate(sky, azimuth_canyon, azimuth_solar, zenith_solar, tree=((0, 0), 0)):
    '''
    Parameters
    ----------
    sky  天空的两个端点坐标
    tree 树冠中心点的坐标和树冠半径
    k    光线的斜率
    azimuth_canyon 街道峡谷的方位角, 弧度制
    azimuth_solar  太阳光线的方位角，弧度制
    Returns 计算建筑树木在街道峡谷表面生成的阴影的坐标值其中（x0,y0）为建筑遮阴坐标，（x1,y1）(x2,y2)为树木遮阴的坐标
    -------
    '''
    # assert type(sky) is tuple and len(sky) == 2 and all(type(sky[x]) is tuple for x in np.arange(0, 2)), '请输入sky的起点和终点坐标以下形式:((x1,y1),(x2,y2))'
    # assert type(tree) is tuple and len(tree) == 2 and type(tree[0]) is tuple, '请输入树冠的圆心和半径以下形式:((x3,y3),r)'
    x1, y1 = sky[0][0], sky[0][1]  # y1和y2为墙面高度
    x2, y2 = sky[1][0], sky[1][1]
    x3, y3, r = tree[0][0], tree[0][1], tree[1]
    x_wall = max(x1, x2)  # x1和x2中大的值为街道宽度
    # 先求光线的当量天顶角：
    angle = np.arctan(np.tan(zenith_solar) * np.sin(azimuth_solar - azimuth_canyon))
    slope = np.tan(np.pi / 2 - angle)
    if azimuth_canyon < azimuth_solar < azimuth_canyon + np.pi:
        '''step1 求建筑遮阴'''
        # 过建筑右侧顶点的直线方程为：y = k (x - w) + h，直线与x和y轴正半轴的交点为墙面遮阴
        x0 = max(0, x_wall - y1 / slope)
        y0 = max(0, y1 - x_wall * slope)
        '''step2 求树木遮阴'''
        # 根据树冠的中线点（x3,y3）和太阳光线的斜率k求过树冠中线点的直线方程：y-y3 = k(x-x3)
        # 上切线的直线方程为：y = k(x - x3 + abs(r * cos(angle))) + y3 + abs(r * sin(angle))
        # 下切线的直线方程为：y = k(x - x3 - abs(r * cos(angle))) + y3 - abs(r * sin(angle))
        # 求上切线与x和y轴正半轴的交点坐标（x1, y1）
        x4 = max(0, (- r * np.abs(np.sin(angle)) - y3) / slope + x3 - r * np.abs(np.cos(angle)))
        y4 = max(0, (r * np.abs(np.cos(angle)) - x3) * slope + y3 + r * np.abs(np.sin(angle)))
        # 求下切线与x和y轴正半轴的交点坐标（x2, y2）
        x5 = max(0, x3 + r * np.abs(np.cos(angle)) + (r * np.abs(np.sin(angle)) - y3) / slope)
        y5 = max(0, - r * np.abs(np.sin(angle)) - y3 - (x3 + r * np.abs(np.cos(angle))) * slope)
        wallshade, treeshade = [[x0, y0], [x_wall, y1]], [[x4, y4], [x5, y5]]
    else:
        '''step1 求建筑遮阴'''
        # 过建筑右侧顶点的直线方程为：y = kx + h，直线与x和x=x_wall的交点为墙面遮阴
        x0 = min(x_wall, - y1 / slope)
        y0 = max(0, y1 + x_wall * slope)
        # 上切线的直线方程为：y = k(x - x3 - r * abs(cos(angle))) + y3 + r * abs(sin(angle))
        # 下切线的直线方程为：y = k(x - x3 + r * abs(cos(angle))) + y3 - r * abs(sin(angle))
        # 求上切线与x和y轴正半轴的交点坐标（x1, y1）
        x4 = min(x_wall, (- r * np.abs(np.sin(angle)) - y3) / slope + x3 + r * np.abs(np.cos(angle)))
        y4 = max(0, slope * (x_wall - x3 - r * np.abs(np.cos(angle))) + y3 + r * np.abs(np.sin(angle)))
        # 求下切线与x和y轴正半轴的交点坐标（x2, y2）
        x5 = min(x_wall, x3 - r * np.abs(np.cos(angle)) + (r * np.abs(np.sin(angle)) - y3) / slope)
        y5 = max(0, slope * (x_wall - x3 + r * np.abs(np.cos(angle))) + y3 - r * np.abs(np.sin(angle)))
        wallshade, treeshade = [[0, y1], [x0, y0]], [[x5, y5], [x4, y4]]
    return wallshade, treeshade


def shade_mat(wall: list, shadecor: list):
    '''
    Parameters
    ----------
    wall                  街道峡谷从左到右的分层列表[0,3][0,2][0,1][0,0][1,0][2,0][3,0][3,1][3,2][3,3]
    shadecordiante        阴影区的起点和终点坐标，树冠：(x1,y1)(x2,y2) 墙面阴影要看遮阴方向
    Returns               街道峡谷各表面的分层列表，[0, 0, 0.5, 1, 1, 1]表示前两个面没有遮阴，第三个面
                          部分遮阴，后三个面完全遮阴
    -------
    '''
    shade_mat = [0] * (len(wall) - 1)
    shade_index, start, final = [], [], []
    for i in np.arange(0, len(wall) - 1):
        if (wall[i][0] <= shadecor[0][0] <= wall[i + 1][0] or wall[i][0] >= shadecor[0][0] >=
            wall[i + 1][0]) and \
                (wall[i][1] <= shadecor[0][1] <= wall[i + 1][1] or wall[i][1] >= shadecor[0][1] >=
                 wall[i + 1][1]):
            start.append(wall.index(wall[i]))
    shade_index.append(start[-1])
    for i in np.arange(0, len(wall) - 1):
        if (wall[i][0] <= shadecor[1][0] <= wall[i + 1][0] or wall[i][0] >= shadecor[1][0] >=
            wall[i + 1][0]) and \
                (wall[i][1] <= shadecor[1][1] <= wall[i + 1][1] or wall[i][1] >= shadecor[1][1] >=
                 wall[i + 1][1]):
            final.append(wall.index(wall[i]))
    shade_index.append(final[0])
    # 先求不考虑阴影长度比例时的遮阴
    shade_mat[shade_index[0]:shade_index[1] + 1] = [1] * (shade_index[1] - shade_index[0] + 1)
    # 求起点所在的面阴影占面长度的比例
    wall_length = np.sqrt(np.power((wall[shade_index[0]][0] - wall[shade_index[0] + 1][0]), 2) \
                          + np.power((wall[shade_index[0]][1] - wall[shade_index[0] + 1][1]), 2))
    shade_length = np.sqrt(np.power((shadecor[0][0] - wall[shade_index[0] + 1][0]), 2) \
                           + np.power((shadecor[0][1] - wall[shade_index[0] + 1][1]), 2))
    frac_1 = shade_length / wall_length
    shade_mat[shade_index[0]] = frac_1
    # 求终点点所在的面阴影占面长度的比例
    wall_length = np.sqrt(np.power((wall[shade_index[1]][0] - wall[shade_index[1] + 1][0]), 2) \
                          + np.power((wall[shade_index[1]][1] - wall[shade_index[1] + 1][1]), 2))
    shade_length = np.sqrt(np.power((wall[shade_index[1]][0] - shadecor[1][0]), 2) \
                           + np.power((wall[shade_index[1]][1] - shadecor[1][1]), 2))
    frac_2 = shade_length / wall_length
    shade_mat[shade_index[1]] = frac_2
    shade_mat = np.array(shade_mat)
    return shade_mat


if __name__ == '__main__':
    a = shadecoordinate(sky=([0, 3], [3, 3]),  azimuth_canyon=0, azimuth_solar=np.pi / 2, zenith_solar= np.pi / 4)
    print(a[0])
    wall = [[0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3]]
    shadecor = [[0, 0], [3, 2.1]]
    a = shade_mat(wall, shadecor)
    print(a)
