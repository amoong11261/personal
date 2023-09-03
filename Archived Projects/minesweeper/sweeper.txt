import pyautogui
import numpy as np
import pandas as pd
import scipy.optimize
import time
from scipy.linalg import lu


class minesweeper:
    def __init__(self):
        self.minelist = []
        self.click_z = True
        self.x, self.y = pyautogui.locateCenterOnScreen('Capture.PNG')
        self.map_size = [16, 30]
        self.map_shown = np.zeros([16, 30])

        self.color_dict = {'192192192': 0, '00255': 1, '01280': 2, '25500': 3, '00128': 4, '12800': 5,
                           '0128128': 6, '000': 7, '128128128': 8}

        name_list = [str(i) + ' ' + str(j) for i in range(16) for j in range(30)]
        self.map_info = pd.DataFrame(np.zeros((16 * 30, 16 * 30 + 1)),
                                     columns=pd.Index(name_list + ['mines']), index=pd.Index(name_list))

        self.map_info.mines = np.NaN

        for i in range(16):
            for j in range(30):
                for points in self.corner_list(i, j):
                    self.map_info.loc[str(i) + ' ' + str(j), str(points[0]) + ' ' + str(points[1])] = 1

        self.record = 0
        self.all_mines_list = []

        self.step = 0

    def reset(self):
        self.minelist = []
        self.click_zero = True
        self.x, self.y = pyautogui.locateCenterOnScreen('Capture.PNG')
        self.map_size = [16, 30]
        self.map_shown = np.zeros([16, 30])

        self.color_dict = {'192192192': 0, '00255': 1, '01280': 2, '25500': 3, '00128': 4, '12800': 5,
                           '0128128': 6, '000': 7, '128128128': 8}

        name_list = [str(i) + ' ' + str(j) for i in range(16) for j in range(30)]
        self.map_info = pd.DataFrame(np.zeros((16 * 30, 16 * 30 + 1)),
                                     columns=pd.Index(name_list + ['mines']), index=pd.Index(name_list))

        self.map_info.mines = np.NaN

        for i in range(16):
            for j in range(30):
                for points in self.corner_list(i, j):
                    self.map_info.loc[str(i) + ' ' + str(j), str(points[0]) + ' ' + str(points[1])] = 1

        self.record = 0
        self.all_mines_list = []

        self.step = 0

    def corner_list(self, m, n):
        s = []
        if m - 1 >= 0:
            s.append([m - 1, n])
        if m + 1 < self.map_size[0]:
            s.append([m + 1, n])
        if n - 1 >= 0:
            s.append([m, n - 1])
        if n + 1 < self.map_size[1]:
            s.append([m, n + 1])
        if (m - 1 >= 0) & (n - 1 >= 0):
            s.append([m - 1, n - 1])
        if (m + 1 < self.map_size[0]) & (n - 1 >= 0):
            s.append([m + 1, n - 1])
        if (m - 1 >= 0) & (n + 1 < self.map_size[1]):
            s.append([m - 1, n + 1])
        if (m + 1 < self.map_size[0]) & (n + 1 < self.map_size[1]):
            s.append([m + 1, n + 1])
        return s

    def read_map(self):
        if not self.click_zero:
            return 0
        im = pyautogui.screenshot()

        for i in range(16):
            for j in range(30):
                color = im.getpixel((self.x - 232 + 16 * j, self.y + 34 + 16 * i))
                color_str = str(color[0]) + str(color[1]) + str(color[2])
                if color_str == '192192192':
                    color = im.getpixel((self.x - 232 + 16 * j + 7, self.y + 34 + 16 * i + 7))
                    color_str = str(color[0]) + str(color[1]) + str(color[2])
                    if color_str == '128128128':
                        self.map_shown[i][j] = 9
                    else:
                        self.map_shown[i][j] = 0
                else:
                    self.map_shown[i][j] = self.color_dict[color_str]
        for pts in self.minelist:
            self.map_shown[pts[0]][pts[1]] = -1

        self.click_zero = False
        return 0

    def click_not_mine(self, point_x, point_y):
        if pyautogui.pixel(int(self.x), int(self.y + 1)) != (255, 255, 0):
            raise TypeError("game failed")
        elif pyautogui.pixel(int(self.x - 4), int(self.y - 4)) != (255, 255, 0):
            raise TypeError("game successed")
        position_str = str(point_x) + ' ' + str(point_y)

        if position_str not in self.map_info.columns:
            return

        if (pyautogui.pixel(int(self.x - 232 + 16 * point_y), int(self.y + 34 + 16 * point_x)) == (192, 192, 192)) & (
                pyautogui.pixel(int(self.x - 232 + 16 * point_y + 7), int(self.y + 34 + 16 * point_x + 7)) == (
                128, 128, 128)):
            pyautogui.click(self.x - 232 + 16 * point_y, self.y + 34 + 16 * point_x)
        color = pyautogui.pixel(int(self.x - 232 + 16 * point_y), int(self.y + 34 + 16 * point_x))
        color_str = str(color[0]) + str(color[1]) + str(color[2])

        if self.color_dict[color_str] != 0:
            mines_around = len(np.intersect1d(self.all_mines_list,
                                              [str(x) + ' ' + str(y) for x, y in self.corner_list(point_x, point_y)]))
            self.map_info.loc[position_str, 'mines'] = self.color_dict[color_str] - mines_around

            self.map_info = self.map_info.drop(position_str, axis=1, errors='ignore')
            return
        else:

            self.map_info = self.map_info.drop(position_str, axis=1, errors='ignore')
            self.map_info = self.map_info.drop(position_str, axis=0, errors='ignore')
            for i in self.corner_list(point_x, point_y):
                if str(i[0]) + ' ' + str(i[1]) in self.map_info.columns:
                    self.click_not_mine(i[0], i[1])

        return

    def solve_first_cell(self):
        test_times = 0
        test_list = [[15, 29], [0, 29], [15, 0], [0, 0],
                     [8, 0], [0, 15], [8, 29], [15, 15],
                     [15, 21], [15, 8], [0, 21], [0, 8],
                     [4, 0], [12, 0], [4, 29], [12, 29]]

        self.click_not_mine(test_list[test_times][0], test_list[test_times][1])
        test_times += 1

        while len(self.map_info.dropna()) == test_times:
            self.click_not_mine(test_list[test_times][0], test_list[test_times][1])
            test_times += 1
        self.step += 1

    def safe_corner(self):
        test_list = [[15, 29], [0, 29], [15, 0], [0, 0]]
        for cell in test_list:
            if (pyautogui.pixel(int(self.x - 232 + 16 * cell[1]), int(self.y + 34 + 16 * cell[0])) == (
                    192, 192, 192)) & (pyautogui.pixel(int(self.x - 232 + 16 * cell[1] + 7), int(
                    self.y + 34 + 16 * cell[0] + 7)) == (128, 128, 128)):
                if str(cell[0]) + ' ' + str(cell[1]) not in self.minelist:
                    return cell
        return 0

    def minesweeper_solver(self):
        if pyautogui.pixel(int(self.x), int(self.y + 1)) != (255, 255, 0):
            raise TypeError("game failed")

        self.record = self.map_info

        if self.map_info.dropna().empty:
            self.solve_first_cell()
            return

        if any(self.map_info.dropna().copy().mines == 0):
            zero_around_cells = self.map_info.dropna().copy()[self.map_info.dropna().copy().mines == 0]
            zero_list = [zero_around_cells.columns[y] for x, y in zip(*np.where(zero_around_cells.values == 1))]
            # print('zero_list', set(zero_list))
            for cell in set(zero_list):
                self.click_not_mine(int(cell.split(' ')[0]), int(cell.split(' ')[1]))

            for index in zero_around_cells.index:
                self.map_info = self.map_info.drop(index, axis=0, errors='ignore')  # drop row
            return

        if len(self.map_info.columns) > 50:

            matrix_to_solve = self.map_info.dropna().copy()
            matrix_to_solve = matrix_to_solve.loc[:, (matrix_to_solve != 0).any(axis=0)].copy()
        else:
            matrix_to_solve = self.map_info.copy()
            matrix_to_solve.loc['all', :] = 1
            matrix_to_solve.loc['all', 'mines'] = 99 - len(self.all_mines_list)
            matrix_to_solve = matrix_to_solve.dropna()

        mine_list_df = matrix_to_solve[matrix_to_solve.iloc[:, :-1].sum(axis=1) == matrix_to_solve.iloc[:, -1]].iloc[:,
                       :-1]
        mine_list = [mine_list_df.columns[y] for x, y in zip(*np.where(mine_list_df.values == 1))]
        if mine_list != []:
            for cell in set(mine_list):
                self.map_info.loc[self.map_info[self.map_info[cell] == 1].index, 'mines'] -= 1
                self.map_info = self.map_info.drop(cell, axis=1, errors='ignore')
                self.map_info = self.map_info.drop(cell, axis=0, errors='ignore')

                self.all_mines_list.append(cell)
            return

        # mines_probability = np.linalg.lstsq(matrix_to_solve.iloc[:, :-1], matrix_to_solve.iloc[:, -1], rcond=None)[0]
        # mines_probability = scipy.optimize.lsq_linear(matrix_to_solve.iloc[:, :-1], matrix_to_solve.iloc[:, -1],
        #                                             bounds=(0, 1), tol = 10**-5)

        mines_probability = scipy.optimize.lsq_linear(matrix_to_solve.iloc[:, :-1], matrix_to_solve.iloc[:, -1],
                                                      bounds=(0, 1), tol=10 ** -5).x + scipy.optimize.lsq_linear(
            matrix_to_solve.iloc[:, :-1], matrix_to_solve.iloc[:, -1], bounds=(0, 1),
            tol=10 ** -8).x + scipy.optimize.lsq_linear(matrix_to_solve.iloc[:, :-1], matrix_to_solve.iloc[:, -1],
                                                        bounds=(0, 1), tol=10 ** -10).x

        mines_probability = np.round(mines_probability, 5)
        not_mine_list = matrix_to_solve.columns[:-1][mines_probability == 0]

        mine_list = matrix_to_solve.columns[:-1][mines_probability == 3]
        # print(not_mine_list, mine_list)

        for cell in not_mine_list:
            self.click_not_mine(int(cell.split(' ')[0]), int(cell.split(' ')[1]))

        for cell in mine_list:
            self.map_info.loc[self.map_info[self.map_info[cell] == 1].index, 'mines'] -= 1

            self.map_info = self.map_info.drop(cell, axis=1, errors='ignore')
            self.map_info = self.map_info.drop(cell, axis=0, errors='ignore')

            self.all_mines_list.append(cell)

        if not_mine_list.empty & mine_list.empty:
            click_success_probability = 1 - mines_probability.min()
            mine_success_probability = 1 - (3 - mines_probability.max()) / 3

            safe_corner_cell = self.safe_corner()
            if (click_success_probability <= 0.5) & (mine_success_probability <= 0.5) & (safe_corner_cell != 0):
                self.click_not_mine(safe_corner_cell[0], safe_corner_cell[1])


            else:

                if click_success_probability > mine_success_probability:  # click a safety cell
                    safest_cell_index = np.where(mines_probability == mines_probability.min())[0][0]
                    safest_cell = matrix_to_solve.columns[:-1][safest_cell_index]

                    self.click_not_mine(int(safest_cell.split(' ')[0]), int(safest_cell.split(' ')[1]))
                    # print(mines_probability.min())
                else:  # add a mine

                    safest_mine_index = np.where(mines_probability == mines_probability.max())[0][0]
                    safest_mine = matrix_to_solve.columns[:-1][safest_mine_index]

                    self.map_info.loc[self.map_info[self.map_info[safest_mine] == 1].index, 'mines'] -= 1

                    self.map_info = self.map_info.drop(safest_mine, axis=1, errors='ignore')
                    self.map_info = self.map_info.drop(safest_mine, axis=0, errors='ignore')

                    self.all_mines_list.append(safest_mine)


        return


if __name__ == "__main__":
    success = 0
    tot = 0
    fail = 0
    fail_at_first = 0
    fail_not_at_first = 0
    a = minesweeper()
    for i in range(100000):
        print(
            f"tot={tot} success={success} fail={fail} fail_at_first={fail_at_first} fail_not_at_first={fail_not_at_first}")
        a.reset()
        for j in range(300):
            try:
                a.minesweeper_solver()
            except TypeError as e:
                print(e)
                if str(e) == "game failed":
                    tot += 1
                    fail += 1
                    time.sleep(1)
                    pyautogui.click(a.x, a.y)
                    time.sleep(1)

                    if a.step == 0:
                        fail_at_first += 1
                    if a.step == 1:
                        fail_not_at_first += 1

                    break
                if str(e) == "game successed":
                    tot += 1
                    success += 1
                    time.sleep(1)
                    pyautogui.click(int(a.x + 10), int(a.y + 200))
                    time.sleep(1)
                    pyautogui.click(int(a.x + 215), int(a.y - 130))
                    time.sleep(1)
                    pyautogui.click(int(a.x), int(a.y))
                    time.sleep(1)
                    break
