import numpy as np
def lineviewfactor(line1, line2):
    '''
    Parameters 需要按照一定的顺序输入参数，顺时针或者逆时针 x1,x2,x3,x4顺时针或者逆时针
    ----------
    line1   表面A的起点和终点坐标
    line2   表面B的起点和终点坐标
    Returns 没有树木遮挡的情况下表面A与表面B的角系数
    -------
    '''
    # assert type(line1) is tuple and len(line1) == 2 and all(type(line1[x]) is tuple for x in range(0, 2)),  '请输入面1的起点和终点坐标以下形式:((x1,y1),(x2,y2))'
    # assert type(line2) is tuple and len(line2) == 2 and all(type(line2[x]) is tuple for x in range(0, 2)),  '请输入面2的起点和终点坐标以下形式:((x3,y3),(x4,y4))'
    x1, y1 = line1[0][0], line1[0][1]
    x2, y2 = line1[1][0], line1[1][1]
    x3, y3 = line2[0][0], line2[0][1]
    x4, y4 = line2[1][0], line2[1][1]
    # 求没有树存在时两个面的角系数，采用交叉线法
    jiaocha1 = np.sqrt(np.square(x1 - x3) + np.square(y1 - y3))
    jiaocha2 = np.sqrt(np.square(x2 - x4) + np.square(y2 - y4))
    feijiaocha1 = np.sqrt(np.square(x1 - x4) + np.square(y1 - y4))
    feijiaocha2 = np.sqrt(np.square(x2 - x3) + np.square(y2 - y3))
    fenmu = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    viewfactor_notree = (jiaocha1 + jiaocha2 - feijiaocha1 - feijiaocha2) / (2 * fenmu)
    return viewfactor_notree

def treeviewfactor(sky, tree=((0, 0), 0)):
    '''
    Parameters
    ----------
    sky  天空的起点和终点坐标，起点和终点没有顺序可言
    tree 需要树冠的半径和树冠中心点坐标

    Returns 计算圆形树冠与天空的角系数，本代码用的计算公式可在大论文第4章找到
    -------
    '''
    # assert type(sky) is tuple and len(sky) == 2 and all(type(sky[x]) is tuple for x in range(0, 2)),  '请输入sky的起点和终点坐标以下形式:((x1,y1),(x2,y2))'
    # assert type(tree) is tuple and len(tree) == 2 and type(tree[0]) is tuple,  '请输入树冠的圆心和半径以下形式:((x3,y3),r)'
    x1, y1 = sky[0][0], sky[0][1]
    x2, y2 = sky[1][0], sky[1][1]
    x3, y3, r = tree[0][0], tree[0][1], tree[1]
    width_road = np.abs(x2 - x1)
    width_tree_1, width_tree_2 = np.abs(x3 - x1), np.abs(x3 - x2)
    h, ht = y1, y3
    view_sky1_t = r / width_tree_1 * np.arctan(width_tree_1 / (h - ht))
    view_sky2_t = r / width_tree_2 * np.arctan(width_tree_2 / (h - ht))
    viewfactor_tree = (width_tree_1 * view_sky1_t + width_tree_2 * view_sky2_t) / width_road
    return viewfactor_tree

if __name__ == '__main__':
    # a = lineviewfactor(((1, 0), (2, 0)), ((10, 0), (10, 2)))
    # print(a)
    b = treeviewfactor(((0, 10), (10, 10)), ((5, 5), 5))
    print(b)




