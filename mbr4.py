import enum
import tkinter as tk
from tkinter import *
import tkinter.messagebox
import numpy as np

p = 'p'
m = 'm'
o = 'o'
fi = 'fi'
di = 'di'
eq = 'eq'
s = 's'
d = 'd'
f = 'f'
si = 'si'
oi = 'oi'
pi = 'pi'
mi = 'mi'
ALL_ATOMS = {p, m, o, fi, di, eq, s, d, f, si, oi, pi, mi}

# 表3-2 基于MBR的主方向关系
# Page 50 表3-2
N = 'N'
S = 'S'
W = 'W'
E = 'E'
B = 'B'
NW = 'NW'
NE = 'NE'
SW = 'SW'
SE = 'SE'
ALL = {N, S, W, E, B, NW, NE, SW, SE}


# 表3-3
class ATOMTable:
    # 行列名
    COLUMNS = [p, m, o, s, d, f, eq, fi, di, si, oi, mi, pi]
    TABLE = [
        [{p}, {p}, {p}, {p}, {p, m, o, d, s}, {p, m, o, d, s}, {p}, {p}, {p}, {p}, {pi}, {p, m, o, d, s}, ALL_ATOMS],
        [{p}, {p}, {p}, {m}, {o, d, s}, {o, d, s}, {m}, {p}, {p}, {m}, {o, d, s}, {f, fi, eq}, {pi, oi, mi, di, si}],
        [{p}, {p}, {p, m, o}, {o}, {o, d, s}, {o, d, s}, {o}, {p, m, o}, {p, m, o, di, fi}, {o, di, fi},
         {o, oi, d, di, s, si, f, fi, eq}, {oi, di, si}, {pi, oi, mi, di, si}],
        [{p}, {p}, {p, m, o}, {s}, {d}, {d}, {s}, {p, m, o}, {p, m, o, di, fi}, {s, si, eq}, {oi, d, f}, {mi}, {pi}],
        [{p}, {p}, {p, m, o, d, s}, {d}, {d}, {d}, {d}, {p, m, o, d, s}, ALL_ATOMS, {pi, oi, mi, d, f},
         {pi, oi, mi, d, f}, {pi}, {pi}],
        [{p}, {m}, {o, d, s}, {d}, {d}, {f}, {f}, {f, fi, eq}, {pi, oi, mi, di, si}, {pi, mi, oi}, {pi, mi, oi}, {pi},
         {pi}],
        [{p}, {m}, {o}, {s}, {d}, {f}, {eq}, {fi}, {di}, {si}, {oi}, {mi}, {pi}],
        [{p}, {m}, {o}, {o}, {o, d, s}, {f, fi, eq}, {fi}, {fi}, {di}, {di}, {oi, di, si}, {oi, di, si},
         {pi, oi, mi, di, si}],
        [{p, m, o, di, fi}, {o, di, fi}, {o, di, fi}, {o, di, fi}, {o, oi, d, di, s, si, f, fi, eq}, {oi, di, si},
         {di}, {di}, {di}, {o, di, si}, {o, di, si}, {o, di, si}, {pi, oi, mi, di, si}],
        [{p, m, o, di, fi}, {o, di, fi}, {o, di, fi}, {s, si, eq}, {oi, d, f}, {oi}, {si}, {di}, {di}, {si},
         {oi}, {mi}, {pi}],
        [{p, m, o, di, fi}, {o, di, fi}, {o, di, fi}, {s, si, eq}, {oi, d, f}, {oi}, {oi}, {oi, di, si},
         {pi, oi, mi, di, si}, {pi, mi, oi}, {oi}, {pi}, {pi}],
        [{p, m, o, di, fi}, {s, si, eq}, {oi, d, f}, {oi, d, f}, {oi, d, f}, {mi}, {mi}, {mi}, {pi}, {pi}, {pi}, {mi},
         {pi}, ],
        [ALL_ATOMS, {pi, oi, mi, d, f}, {pi, oi, mi, d, f}, {pi, oi, mi, d, f}, {pi, oi, mi, d, f}, {pi}, {pi},
         {pi}, {pi}, {pi}, {pi}, {pi}, {pi}, ]
    ]

    @classmethod
    def get_op_result(cls, s1, s2):
        ...


class MBRTable:
    # 行列名
    COLUMNS = {
        0: {p, m},
        1: {o, fi},
        2: {di},
        3: {eq, s, d, f},
        4: {si, oi},
        5: {pi, mi},
    }
    # 表格
    TABLE = [
        [{SW}, {SW, S}, {SW, S, SE}, {S}, {S, SE}, {SE}],
        [{W, SW}, {W, B, SW, S}, {W, B, E, SW, S, SE}, {B, S}, {B, E, S, SE}, {E, SE}],
        [{NW, W, SW}, {NW, N, W, B, SW, S}, ALL, {N, B, W}, {N, NE, B, E, S, SE}, {NE, E, SE}],
        [{W}, {W, B}, {W, B, E}, {B}, {B, E}, {E}],
        [{NW, W}, {NW, N, W, B}, {NW, N, NE, W, B, E}, {N, B}, {N, NE, B, E}, {NE, E}],
        [{NW}, {NW, N}, {NW, N, NE}, {N}, {N, NE}, {NE}]
    ]

    @classmethod
    def get_relationship(cls, direction):
        assert isinstance(direction, set)
        n = 6
        for i in range(n):
            for j in range(n):
                if cls.TABLE[i][j] == direction:
                    return cls.COLUMNS[j], cls.COLUMNS[i]
        raise ValueError('方向关系 {} 未找到'.format(direction))

    @classmethod
    def get_queries(cls, s1, s2):
        values = list(cls.COLUMNS.values())
        i1 = values.index(s1)
        i2 = values.index(s2)
        return cls.TABLE[i2][i1]

    @classmethod
    def get_inter(cls, operations):
        result = []
        for row in cls.COLUMNS.values():
            op = row & operations
            if op == row:
                result.append(op)
        return result

    @classmethod
    def get_directions(cls, s1, s2):
        result = []
        for s1i in s1:
            for s2j in s2:
                print('===>', s1i, s2j, cls.get_queries(s1i, s2j))
                result.append(cls.get_queries(s1i, s2j))
        return result


# 表3-2
class BaseIntervalComb:
    COLUMNS = [{p, m}, {o, fi}, {di}, {eq, s, d, f}, {si, oi}, {pi, mi}]
    TABLE = [
        [{p, m}, {p, m}, {p, m}, {p, m, o, d, s}, {p, m, o, d, s}, ALL_ATOMS],
        [{p, m}, {p, m, o, fi}, {p, m, o, fi, di}, {o, fi, eq, s, d, f}, {o, fi, di, eq, s, d, f, si, oi},
         {di, si, oi, pi, mi}],
        [{p, m, o, fi, di}, {o, fi, di}, {di}, {o, fi, di, eq, s, d, f, si, oi}, {di, oi, si}, {di, si, oi, pi, mi}],
        [{p, m}, {p, m, o, fi, eq, s, d, f}, ALL_ATOMS, {eq, s, d, f}, {eq, s, d, f, oi, si, pi, mi}, {pi, mi}],
        [{p, m, o, fi, di}, {o, fi, di, eq, s, d, f, si, oi}, {di, oi, si, pi, mi, },
         {eq, s, d, f, oi, si,}, {oi, si, pi, mi}, {pi, mi}],
        [ALL_ATOMS, {d, f, oi, pi, mi}, {pi, mi}, {d, f, oi, pi, mi}, {pi, mi}, {pi, mi}],
    ]

    @classmethod
    def get_result(cls, r1, r2):
        i1 = cls.COLUMNS.index(r1)
        i2 = cls.COLUMNS.index(r2)
        return cls.TABLE[i2][i1]


def calculate_comb_result(d1, d2):
    d11, d12 = MBRTable.get_relationship(d1)
    d21, d22 = MBRTable.get_relationship(d2)
    s1, s2 = BaseIntervalComb.get_result(d11, d22), BaseIntervalComb.get_result(d12, d21)
    s1 = MBRTable.get_inter(s1)
    s2 = MBRTable.get_inter(s2)

    return MBRTable.get_directions(s1, s2)


def test_get_relationship():
    got = MBRTable.get_relationship({NW})
    expect = ({p, m}, {pi, mi})
    assert got == expect

    got = MBRTable.get_relationship({W})
    expect = ({p, m}, {s, eq, d, f})
    assert got == expect


def test_atom_table():
    n = len(ATOMTable.COLUMNS)
    for i, row in enumerate(ATOMTable.TABLE):
        assert n == len(row), (i, ATOMTable.TABLE[i])


def test_direction_relationship():
    d1 = {W}
    d2 = {B, S, SW, W}

    d11, d12 = MBRTable.get_relationship(d1)
    d21, d22 = MBRTable.get_relationship(d2)
    s1, s2 = BaseIntervalComb.get_result(d11, d22), BaseIntervalComb.get_result(d12, d21)
    print(d11, d12)
    print(d21, d22)
    print(s1, s2)
    s1 = MBRTable.get_inter(s1)
    s2 = MBRTable.get_inter(s2)
    print(s1)
    print(s2)

    print(MBRTable.get_directions(s1, s2))


def test_direction_relationship2():
    d1 = {N}
    d2 = {S, SW}

    d11, d12 = MBRTable.get_relationship(d1)
    d21, d22 = MBRTable.get_relationship(d2)
    print(d11, d12)
    print(d21, d22)
    s1 = BaseIntervalComb.get_result(d11, d22)
    s2 = BaseIntervalComb.get_result(d12, d21)

    print(s1, s2)
    s1 = MBRTable.get_inter(s1)
    s2 = MBRTable.get_inter(s2)
    print(s1)
    print(s2)

    print(MBRTable.get_directions(s1, s2))


def test_direction_relationship3():
    d1 = {S}
    d2 = {S, SW}

    d11, d12 = MBRTable.get_relationship(d1)
    d21, d22 = MBRTable.get_relationship(d2)
    s1, s2 = BaseIntervalComb.get_result(d11, d22), BaseIntervalComb.get_result(d12, d21)
    print(d11, d12)
    print(d21, d22)
    print(s1, s2)
    s1 = MBRTable.get_inter(s1)
    s2 = MBRTable.get_inter(s2)
    print(s1)
    print(s2)

    print(MBRTable.get_directions(s1, s2))


def test_direction_relationship4():
    d1 = {NE}
    d2 = {NE, E}

    d11, d12 = MBRTable.get_relationship(d1)
    d21, d22 = MBRTable.get_relationship(d2)
    s1, s2 = BaseIntervalComb.get_result(d11, d22), BaseIntervalComb.get_result(d12, d21)
    print(d11, d12)
    print(d21, d22)
    print(s1, s2)
    s1 = MBRTable.get_inter(s1)
    s2 = MBRTable.get_inter(s2)
    print('s1 = ', s1)
    print('s2 = ', s2)

    print(MBRTable.get_directions(s1, s2))


# W
# B:S:SW:W
def parse_verify(sv):
    sv = sv.get().strip()
    items = sv.split(':')
    print(items)
    if not all(item in ALL for item in items):
        return {}
    return set(items)


def calculate(text_ab, text_bc, result_var):
    d1 = parse_verify(text_ab)
    d2 = parse_verify(text_bc)
    if not d1 or not d2:
        tkinter.messagebox.showerror(title='错误', message='方向输入正确，合法的值为: {}'.format(ALL))
        return
    result = calculate_comb_result(d1, d2)
    result = ', '.join([':'.join(items) for items in result])
    result_var.set('计算结果:' + result)


def main():
    app = tk.Tk()
    app.title('主方向关系推理')
    app.geometry('400x300')
    app.resizable(False, False)

    label = Label(app, text='A和B之间的关系')
    label.grid(row=0, column=0, sticky=N, padx=10, pady=20)
    text_ab = Text(app, width=12, height=2)
    text_ab.grid(row=0, column=1, pady=20, sticky=W)

    label = Label(app, text='B和C之间的关系')
    label.grid(row=1, column=0, sticky=N, padx=10, pady=20)
    text_bc = Text(app, width=12, height=2)
    text_bc.grid(row=1, column=1, pady=20, sticky=W)

    relation_ab = StringVar()
    relation_bc = StringVar()
    text_ab.bind('<KeyRelease>', lambda event: relation_ab.set(text_ab.get("1.0", END)))
    text_bc.bind('<KeyRelease>', lambda event: relation_bc.set(text_bc.get("1.0", END)))

    result_var = StringVar()
    label_result = Label(app, text='计算结果', textvariable=result_var, justify='left')
    label_result.grid(row=2, column=1, sticky=N, pady=20, padx=10)

    button = Button(app, text='开始计算', width=15, command=lambda: calculate(relation_ab, relation_bc, result_var))
    button.grid(row=3, column=0, columnspan=3, sticky=W, padx=10)

    app.mainloop()


if __name__ == '__main__':
    main()
