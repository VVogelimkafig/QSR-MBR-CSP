import tkinter as tk
from tkinter import *
import tkinter.messagebox
from itertools import combinations, product
from copy import deepcopy


def ps(row):
    """解析每一行"""
    items = row.split()
    result = []
    for item in items:
        result.append(set(item.split(',')))
    assert len(result) == 14
    return result


def collect_circles(graph):
    """
    给定一个(连通的)无向图，找到所有的环
    :param graph:
    :return:
    """
    nodes = set(graph)
    visited = set()
    # 记录已经找到的所有环
    cycles = []
    # 任意选择一个节点开始
    start = nodes.pop()
    stack = [(start, [start])]
    while stack:
        cur, path = stack.pop()
        # 若该节点已经被访问过，则跳过
        if cur in visited:
            continue
        # 标记为访问过
        visited.add(cur)
        # 跳转到相邻节点
        for neighbour in graph[cur]:
            # 相邻的节点已经在路径上，而且不是当前节点的直接前驱节点，则说明找到了一个环
            if neighbour in path and neighbour != path[-2]:
                index = path.index(neighbour)
                cycles.append(path[index:])
                continue
            # 未访问过
            elif neighbour not in visited:
                stack.append((neighbour, path + [neighbour]))
    return cycles


# 表4.4 D14 基本组合表:dir(X,Y)∞dir(Y,Z)→ dir(X,Z)
class Table:
    COLUMNS = ['RN', 'NE', 'RE', 'SE', 'RS', 'SW', 'RW', 'NW', 'SA', 'N', 'E', 'S', 'W', 'nil']
    TABLE = [
        ps('RN NE E,NE,RE RE,E,NE,SE nil,SA,RS,RN NW,W,RW,SW NW,RW,W NW  RN,nil,SA NE,RN,NW,N NE,RE,E All W,RW,NW    NE,NW,E,W,N,RN,RE,RW,SA,nil'),
        ps("""NE,RN,N NE NE,RE,E NE,E,RE,SE  NE,E,RE,SE,nil,SA,N,RN,S,RS ALL  N,E,RN,RE,NE,nil,SA,NW,W,RW   RN,N,NE,NW
       NE,SA,nil,RE,E,RN,N NE,N,RN RE,E,NE NE,SE,N,E,S,SA,RN,RE,RS,nil  NE,NW,N,E,W,SA,nil,RN,RE,RW  NE,N,E,RN,RE,SA,nil
       """),
        ps("""NE,N,RN  NE RE SE SE,S,RS SE,RS,S,SW SA,RE,nil,RW NE,RN,N,NW 
        RE,nil,SA NE,RN,N SE,RE,NE,E S,RS,SE ALL N,E,S,RN,RE,RS,NE,SE,SA,nil"""),
        ps("""SE,NE,S,N,E,SA,RS,RN,RE,nil SE,E,RE,NE  SE,E,RE SE SE,S,RS  SE,S,RS,SW   SE,SW,E,S,W,SA,RS,RE,RW,nil
        ALL  SE,SA,nil,RE,E,RS,S  SE,NE,S,N,E,SA,RE,RN,RS,nil  E,SE,RE  SE,RS,S   SE,SW,RS,RE,RW,S,E,W,SA,nil
        SE,S,RS,SA,nil,E,RE
    """),
        ps("""RS,RN,SA,nil  SE,E,RE,NE  SE,RE,E SE RS SW SW,W,RW SW,W,RW,NW   RS,SA,nil ALL  SE,E,RE  S,RS,SE,SW
        W,SW,RW S,RS,SW,nil,SE,SA,RE,RW,E,W
    """),
        ps("""SW,NW,nil,RW,S,RS,N,RN,SA,W  ALL  S,W,E,RS,RW,SA,RE,SW,SE,nil  S,RS,SE,NS  SW,S,RS  SW  SW,W,RW   SW,W,RW,NW
    SW,S,RS,nil,SA,RW,W  SW,NW,S,W,N,SA,nil,RS,RW,RN SW,SE,S,W,E,SA,nil,RS,RW,RE RS,S,SW SW,W,RW  
    SE,S,RS,SA,nil,E,RE

    """),
        ps("""NW,N,RN NW,N,RN,NE RE,RW,nil,SA SW,S,RS,SE  S,SW,RS SW RW NW 
    RW,nil,SA NW,N,RN  ALL SW,S,RS SW,W,RW,NW N,S,W,RN,RS,RW,NW,SW,SA,nil
    """),
        ps("""NW,N,RN NW,N,RN,NE NE,NW,N,E,W,RN,RE,RW,SA,nil ALL NW,NE,N,W,E,SA,RN,RW,RE,nil W,NW,SW,RW
    NW,RW,W NW  NW,SA,nil,W,N,RN,RW  N,NW,RN  NW,SA,nil,RW,N,RN,W,NE,RE,E   SA,nil,NW,W,RW,RN,N,S,RS,SW
    W,NW,RW  NW,SA,RN,RW,nil,W,N,
    """),
        ps("""RN NE RE SE RS SW RW NW SA RN,NE,NW,N  RE,NE,SE,E  RS,SE,SW,S  RW,SW,NW,W ALL"""),
        ps("""N,RN NE,N N,nil,E,RE,NE  NE,SE,nil,RE,E,N,S   RS,S,nil,SA,N,RN 
    W,NW,nil,SW,RW,N,S  NW,W,RW,nil,N  NW,N N,RN,SA,nil RN,NW,NE,N RE,N,NE,E,nil  ALL
    RW,N,NW,W,nil  NE,NW,E,W,N,RN,RE,RW,SA,nil
    """),
        ps("""NE,E,N,nil,RN  NE,E RE,E SE,E   SE,E,S,nil,RS  W,nil,S,RS,E,SE,SW  SA,nil,RE,RW,E,W
    NW,nil,NE,N,E,W,RN  E,SA,nil,RE  RN,E,NE,N,nil  RE,NE,SE,E  RS,E,S,SE,nil  ALL  N,E,S,RN,RE,RS,NE,SE,SA,nil

    """),
        ps("""RN,RS,nil,SA,N,S   NE,S,SE,N,E,nil,RE  RE,E,SE,nil,S SE,S  S,RS  S,SW  S,W,SW,RW,nil
    SW,W,RW,NW,N,S,nil   S,RS,SA,nil  ALL  SE,S,E,RE,nil   RS,S,SW,SE  SW,S,W,RW,nil  
    S,RS,SW,nil,SE,SA,RE,RW,E,W
    """),
        ps("""W,NW,RN,N,nil  NE,N,RN,EN,W,nil  E,W,RE,RW,SA,nil  SW,W,S,RS,SE,E,nil  S,RS,SW,W,nil
    SW,W
    W,RW
    NW,W
    W,RW,SA,nil  NW,RN,W,N,nil  ALL  SW,RS,S,W,nil  RW,SW,NW,W   RN,RW,RS,N,S,W,NW,SW,SA,nil
    """),
        ps("""SA,nil,N,RN  NE,E,N,SA,nil   SA,RE,E,nil  SE,S,E,SA,nil  RS,S,SA,nil  SW,S,W,SA,nil 
    RW,W,SA,nil NW,N,W,SA,nil   SA,nil  RN,N,SA,RE,RW,E,W,NW,NE,nil  RE,E,SA,RS,RN,S,N,SE,NE,nil
    RS,S,SA,RE,RW,E,W,SE,SW,nil
    RW,W,SA,RS,RN,S,N,NW,SW,nil
    ALL
    """),
    ]

    @classmethod
    def get_inf_result(cls, r1, r2):
        """
        给定两个关系，返回所有可能的关系集合
        dir(X,Y)∞dir(Y,Z)→ dir(X,Z)
        :param r1: 关系 X,Y
        :param r2: 关系 Y,Z
        :return: XZ可能的关系
        """
        assert isinstance(r1, str) and isinstance(r2, str)
        i1 = cls.COLUMNS.index(r1)
        i2 = cls.COLUMNS.index(r2)
        return cls.TABLE[i1][i2]

    @classmethod
    def build_undirected_graph(cls, net):
        """
        构建无向图
        :return:
        """
        graph = dict()
        for x, y, relation in net:
            for e in (x, y):
                if e not in graph:
                    graph[e] = set()
            graph[x].add(y)
            graph[y].add(x)
        return graph

    @classmethod
    def check_if_fit(cls, net):
        """
        输入几个对象之间的关系，判断关系网络是否合理
        :param net: 列表，每个元素为三元组 (X, Y, D)，表示X和Y之间的关系为D
        :return: 返回True如果满足，返回False如果不满足
        """
        assert net, '关系网不能为空!'
        # step 1, 将其转换为无向图
        graph = cls.build_undirected_graph(net)
        # step 2, 检测环
        cycles = collect_circles(graph)
        nodes_in_cycle = set()
        for cycle in cycles:
            nodes_in_cycle |= set(cycle)
        # Step 3, 在环上的节点构建有向图
        graph = dict()
        for x, y, relation in net:
            if x not in nodes_in_cycle or y not in nodes_in_cycle:
                continue
            if x not in graph:
                graph[x] = dict()
            graph[x][y] = relation
        # Step 4, 判断所有三元组约束是否满足
        # print(graph)
        for x in graph:
            for y in graph:
                if x == y:
                    continue
                # 如果x和y之间存在关系
                if y in graph[x]:
                    # 并且y和z之间存在关系
                    for z in graph[y]:
                        # 并且z和x之间存在关系
                        if z in graph[x]:
                            r1 = graph[x][y]
                            r2 = graph[y][z]
                            gt = graph[x][z]
                            # 查表计算r1和r2之间所有可能的关系
                            possible_relationship = Table.get_inf_result(r1, r2)
                            # 若x和z之间的关系不存在于可能的关系集合，说明约束不满足
                            if gt not in possible_relationship:
                                error_msg = f'三元组({x}, {y}, {z}) 之间的关系不满足约束:\n' + \
                                            f'\t{x}和{y}之间的关系为 {r1}\n' + \
                                            f'\t{y}和{z}之间的关系为 {r2}\n' + \
                                            f'\t{x}和{z}之间的关系为 {gt}, 非法，合法的关系集合为 {possible_relationship}'
                                return False, error_msg

        return True, None

    @classmethod
    def infer(cls, net):
        """
        推理满足约束的组合
        :param net:
        :return:
        """
        # 首先判断是否满足约束
        status, reason = cls.check_if_fit(net)
        if not status:
            return status, reason

        # step 1, 将其转换为无向图
        graph = cls.build_undirected_graph(net)
        all_nodes = set(graph)
        # step 2, 检测环
        cycles = collect_circles(graph)
        print('找到的环:', cycles)
        nodes_in_cycle = set()
        for cycle in cycles:
            nodes_in_cycle |= set(cycle)
        print('环上的节点:', nodes_in_cycle)
        # Step 3, 在环上的节点构建有向图
        graph = dict()
        for x, y, relation in net:
            if x not in graph:
                graph[x] = dict()
            graph[x][y] = relation
        # Step 4, 推理可能的关系组合
        infer_result = []
        for x in graph:
            for y in graph:
                if x == y:
                    continue
                # 如果x和y之间存在关系
                if y in graph[x]:
                    # 并且y和z之间存在关系
                    for z in all_nodes:
                        # z和x之间不存在关系，则可以进行推理
                        if z not in graph[x] and z in graph[y]:
                            r1 = graph[x][y]
                            r2 = graph[y][z]
                            possible_relationship = Table.get_inf_result(r1, r2)
                            # x和z可能存在关系
                            infer_result.append((x, z, possible_relationship))
        return True, infer_result

    @classmethod
    def iterative_infer_dfs_helper(cls, net, relations_to_fill, maximum, best=[]):
        """
        在调用算法之前必须要判断net是否满足约束
        推理满足约束的最多的组合
        :param net: 列表，每个元素为三元组 (X, Y, D)，表示X和Y之间的关系为D
        :param best: 记录最佳结果
        :param relations_to_fill: 还需要生成的关系
        :param maximum: 最大可能的关系数
        :return:
        """
        # 递归终止条件,没有需要补充的关系
        # 或者已经找到所有关系
        if not relations_to_fill or (best and len(best[-1]) >= maximum):
            return
        # 所有可能的关系
        POSSIBLE_RELATIONSHIP = ['RN', 'NE', 'RE', 'SE', 'RS', 'SW', 'RW', 'NW', 'SA', 'N', 'E', 'S', 'W']

        for i, (n1, n2) in enumerate(relations_to_fill):
            for p in POSSIBLE_RELATIONSHIP:
                r = (n1, n2, p)
                # 加入关系后不冲突
                if Table.check_if_fit(net + [r])[0]:
                    tmp = deepcopy(relations_to_fill)
                    tmp.remove((n1, n2))
                    new_solution = net + [r]
                    Table.iterative_infer_dfs_helper(new_solution, tmp, maximum, best)
                    # 找到一个更优(路径更长、约束条件更多)的解
                    if not best or len(new_solution) > len(best[-1]):
                        best.append(new_solution)
                    if best and len(best[-1]) >= maximum:
                        return

    @classmethod
    def iterative_infer_dfs(cls, net):
        """
        找到最大的约束
        :param net:
        :param maximum:
        :return:
        """
        # 检查是否满足约束
        status, msg = cls.check_if_fit(net)
        # 不满足约束
        if not status:
            raise ValueError(msg)
        existed_relationships = set()
        nodes = set()
        for n1, n2, r in net:
            nodes.add(n1)
            nodes.add(n2)
            existed_relationships.add((min(n1, n2), max(n1, n2)))
        relations_to_fill = set()
        for n1, n2 in combinations(nodes, 2):
            n1, n2 = min(n1, n2), max(n1, n2)
            if (n1, n2) not in existed_relationships:
                relations_to_fill.add((n1, n2))
        best = []
        maximum = len(nodes) * (len(nodes) - 1) // 2
        cls.iterative_infer_dfs_helper(net, relations_to_fill, maximum, best)
        result = best[-1]
        print(best)
        return result


def test_table():
    assert len(Table.COLUMNS) == 14, len(Table.COLUMNS)
    for row in Table.TABLE:
        assert len(row) == 14, len(row)
    # W(X,Y)
    r1 = 'W'
    # NW(Y,Z)
    r2 = 'NW'
    # 则X和Z的关系为
    result = Table.get_inf_result(r1, r2)
    print(result)


def test_check_match():
    relation_net = [('商业区', '工矿区', 'NW'), ('生活区', '商业区', 'RN'),
                    ('生活区', '水源', 'E'), ('学校', '生活区', 'RW'), ('运动场馆', '商业区', 'W'),
                    ('运动场馆', '工矿区', 'N')]
    result, reason = Table.check_if_fit(relation_net)
    print('是否满足:', result)
    print(reason)


def test_infer():
    relation_net = [('a', 'b', 'N'), ('b', 'c', 'N')]
    status, infer_result = Table.infer(relation_net)
    print(status, infer_result)


def test_iterative_infer2():
    relation_net = [('生活区', '商业区', 'RN'),
                    ('生活区', '水源', 'E'),
                    ('学校', '生活区', 'RW')]
    # ('运动场馆', '工矿区', 'N')
    solution = Table.iterative_infer_dfs(relation_net)
    print('最优解:', solution)
    print('最优解是否满足约束:', Table.check_if_fit(solution))
    print()


def test_iterative_infer3():
    relation_net = [('A', 'B', 'N'),
                    ('B', 'C', 'E'),
                    ('C', 'D', 'SE'),
                    ('D', 'E', 'N')]
    # ('运动场馆', '工矿区', 'N')
    solution = Table.iterative_infer_dfs(relation_net)
    print('最优解:', solution)
    print('最优解是否满足约束:', Table.check_if_fit(solution))
    print()


"""
示例输入1 该网络不是合法的关系网络
商业区,工矿区,NW
生活区,商业区,RN
生活区,水源,E
学校,生活区,RW
运动场馆,商业区,W
运动场馆,工矿区,N

示例输入2，该网络不是合法的关系网络
A,B,N
B,C,N
A,C,S

示例输入3: 该网络不是合法的关系网络，用于测试求解
A,B,N
B,C,E
C,D,SE
D,E,N
"""


def parse_relation_net(inputs):
    lines = inputs.splitlines()
    relation_net = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        items = line.split(',')
        items = [e.strip() for e in items]
        if len(items) != 3:
            tkinter.messagebox.showerror(title='错误', message='第{}行格式不正确'.format(i + 1))
            return
        if items[-1] not in Table.COLUMNS:
            tkinter.messagebox.showerror(title='错误', message='第{}行关系不正确,合法的关系为 {}'.format(
                i + 1, ','.join(Table.COLUMNS)
            ))
            return
        relation_net.append(tuple(items))
    if not relation_net:
        tkinter.messagebox.showerror(title='错误', message='输入为空')
        return
    return relation_net


def calculate(relation_net_str, ):
    inputs = relation_net_str.get()
    relation_net = parse_relation_net(inputs)
    if not relation_net:
        return
    result, reason = Table.check_if_fit(relation_net)
    s = ''
    if not result:
        s += '网络不满足约束, 原因为: {}'.format(reason)
    else:
        s += '网络满足约束!'
    tkinter.messagebox.showinfo(title='检查结果', message=s)


def infer(relation_net_str):
    inputs = relation_net_str.get()
    relation_net = parse_relation_net(inputs)
    status, reason = Table.check_if_fit(relation_net)
    if not status:
        tkinter.messagebox.showinfo(title='检查结果', message='网络不满足约束, 原因为: {}'.format(reason))
        return
    try:
        solution = Table.iterative_infer_dfs(relation_net)
    except Exception as e:
        tkinter.messagebox.showinfo(title='求解失败', message='原因为: {}'.format(e))
        return
    print('最优解:', solution)
    # 检查最优解是否满足约束，最优解中一定不会有冲突
    status, reason = Table.check_if_fit(solution)
    if not status:
        tkinter.messagebox.showerror(title='算法执行出错', message='最优解中有冲突')
    r = '\n'.join('{},{},{}'.format(*e) for e in solution)
    tkinter.messagebox.showinfo(title='一致性场景', message=r)


def main():
    app = tk.Tk()
    app.title('约束满足检查')
    app.geometry('400x350')
    app.resizable(False, False)

    label = Label(app, text='请输入关系网，每行一个关系，格式为 x,y,r')
    label.grid(row=0, column=0, sticky=N, padx=10, pady=20)
    text_net = Text(app, width=20, height=10)
    text_net.grid(row=1, column=0, pady=0, sticky=W, padx=20)

    relation_net_str = StringVar()
    text_net.bind('<KeyRelease>', lambda event: relation_net_str.set(text_net.get("1.0", END)))

    result_var = StringVar()
    label_result = Label(app, text='计算结果', textvariable=result_var, justify='left')
    label_result.grid(row=2, column=1, sticky=N, pady=20, padx=10)

    button = Button(app, text='检查约束是否满足', width=15, command=lambda: calculate(relation_net_str, ))
    button.grid(row=3, column=0, columnspan=3, sticky=W, padx=10)

    button2 = Button(app, text='推出可能存在的解', width=15, command=lambda: infer(relation_net_str, ))
    button2.grid(row=4, column=0, columnspan=3, sticky=W, padx=10)

    app.mainloop()


if __name__ == '__main__':
    main()
