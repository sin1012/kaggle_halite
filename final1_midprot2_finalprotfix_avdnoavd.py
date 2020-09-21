import time
import copy
import math
from collections import deque, defaultdict
import pprint
import scipy.optimize
import scipy.ndimage
from kaggle_environments.envs.halite.helpers import *
import random
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


import numpy as np
from numpy.lib.stride_tricks import as_strided

print_ = copy.deepcopy(print)
inds = pd.DataFrame([list(range(20, -1, -1)), list(range(21))]).T.set_index([0, 1]).index
inds.names = ['lvl1', 'lvl2']


#################################################################
# utilility functions
#################################################################

def translate_pos(position):
    row = size - 1 - position.y
    col = position.x
    return row, col


def slice(halite, start_row, end_row, start_col, end_col):
    return np.tile(halite, (3, 3))[start_row + size: end_row + size + 1,  # adj9
           start_col + size: end_col + size + 1]


def circular_distance(num1, num2):
    if num1 > num2:
        num1, num2 = num2, num1
    return min(abs(num1 - num2), abs(num1 + size - num2))


def recover_pos(pos, row, col, rr):
    return tuple(((pos + np.array([row, col]) - rr) + size) % size)


def has_my_ship(flag):
    return np.logical_or(flag == 1, flag == 5)


def has_my_shipyard(flag):
    return np.logical_or(flag == 3, flag == 5)


def gen_enemy_halite_matrix(board):
    # generate matrix of enemy positions:
    # EP=presence of enemy ship
    # EH=amount of halite in enemy ships
    # ES=presence of enemy shipyards
    EP = np.zeros((size, size))
    EH = np.zeros((size, size))
    ES = np.zeros((size, size))
    for id, ship in board.ships.items():
        if ship.player_id != me.id:
            EH[ship.position.y, ship.position.x] = ship.halite
            EP[ship.position.y, ship.position.x] = 1
    for id, sy in board.shipyards.items():
        if sy.player_id != me.id:
            ES[sy.position.y, sy.position.x] = 1
    return EP, EH, ES


def dist(a, b):
    # Manhattan distance of the Point difference a to b, considering wrap around
    action, step = dirs_to(a, b, size=21)
    return abs(step[0]) + abs(step[1])


def nearest_shipyard(pos):
    # return distance, position of nearest shipyard to pos.  100,None if no shipyards
    mn = 100
    best_pos = None
    for sy in me.shipyards:
        d = dist(pos, sy.position)
        if d < mn:
            mn = d
            best_pos = sy.position
    return mn, best_pos


def get_ship_ratio(row, col, rr):
    key = (row, col, rr)
    if key in ship_ratios:
        return ship_ratios[key]

    flag = slice(turn.flag_matrix, row - rr, row + rr, col - rr, col + rr)
    my_flag_counts = np.sum(flag == 1)
    enemy_flag_counts = np.sum(flag == 2) + 1e-6
    res = my_flag_counts / enemy_flag_counts
    ship_ratios[key] = res
    return res


def get_cargo_matrix():
    cargo_matrix = np.ones((size, size)) * np.nan
    for ship in board.ships.values():
        row, col = translate_pos(ship.position)
        cargo_matrix[row, col] = ship.halite
    return cargo_matrix


def get_cargo_ratio(row, col, rr):
    key = (row, col, rr)
    if key in cargo_ratios:
        return cargo_ratios[key]

    flag = slice(turn.flag_matrix, row - rr, row + rr, col - rr, col + rr)
    cargo = slice(cargo_matrix, row - rr, row + rr, col - rr, col + rr)
    my_flag_counts = np.sum(flag == 1)
    if my_flag_counts == 0:
        my_avg = 0
    else:
        my_avg = np.sum(cargo * (flag == 1)) / my_flag_counts
    enemy_flag_counts = np.sum(flag == 2)
    if enemy_flag_counts == 0:
        enemy_avg = 10000
    else:
        enemy_avg = (np.sum(cargo * (flag == 2)) / enemy_flag_counts) + 1e-6
    res = my_avg / enemy_avg
    cargo_ratios[key] = res
    return res


def get_min_enemy_cargo(row, col, rr):
    enemy_cargo = slice(cargo_matrix, row - rr, row + rr, col - rr, col + rr)
    enemy_flag = slice(turn.flag_matrix, row - rr, row + rr, col - rr, col + rr) == 2
    enemy_cargo = enemy_cargo * enemy_flag
    if np.sum(enemy_flag) > 0:
        return np.nanmin(enemy_cargo)
    else:
        return 10000


def view_C(C, pts):
    C_df = np.zeros((size, size))
    for cc, pt in zip(C, pts):
        row, col = translate_pos(pt)
        C_df[row, col] = cc
    C_df = pd.DataFrame(C_df, index=inds)
    print_(C_df)
    return C_df


def find_within(targets, row, col, within):
    targets = set(targets)
    q = deque([(row, col, 0)])
    visited = set()
    res = []
    while q:
        curr_row, curr_col, curr_step = q.popleft()
        if (curr_row, curr_col) in visited:
            continue
        visited.add((curr_row, curr_col))
        if turn.flag_matrix[curr_row, curr_col] in targets:
            res.append((curr_step, (curr_row, curr_col)))
        if curr_step > within:
            break
        for next_row, next_col in [(curr_row + 1, curr_col),
                                   (curr_row - 1, curr_col),
                                   (curr_row, curr_col + 1),
                                   (curr_row, curr_col - 1)]:
            if (0 <= next_row <= size - 1 and 0 <= next_col <= size - 1):
                if (next_row, next_col) not in visited:
                    q.append((next_row, next_col, curr_step + 1))

    res = sorted(res)
    return res


def in_middle(mid_point, start_point, end_point):
    # start -> mid
    vec1 = (mid_point.x - start_point.x, mid_point.y - start_point.y)
    # start -> end
    vec2 = (end_point.x - start_point.x, end_point.y - start_point.y)
    # end -> mid
    vec3 = (mid_point.x - end_point.x, mid_point.y - end_point.y)
    # end -> start
    vec4 = (start_point.x - end_point.x, start_point.y - end_point.y)

    cos_sim1 = np.dot(vec1, vec2)
    cos_sim2 = np.dot(vec3, vec4)
    return cos_sim1 > 0 and cos_sim2 > 0


def get_enemies_around_point(board, pt, size=1, is_enemy=True):
    global enemies_around_point
    if (pt, size, is_enemy) in enemies_around_point:
        return enemies_around_point[(pt, size, is_enemy)]

    all_cells = board.cells
    total_enemy = 0
    all_enemies = []
    for ii in list(range(-size, 0)) + [0] + list(range(1, size + 1)):
        curr_max_jj = abs(size - abs(ii))
        for jj in list(range(-curr_max_jj, 0)) + [0] + list(range(1, curr_max_jj + 1)):
            curr_pos = pt + Point(ii, jj)
            curr_pos = curr_pos % 21
            # print(curr_pos)
            curr_cell = all_cells[curr_pos]
            if curr_cell.ship:
                if is_enemy == True:
                    if curr_cell.ship.player_id != me.id:
                        total_enemy += 1
                        all_enemies.append(curr_cell.ship)
                else:
                    if curr_cell.ship.player_id == me.id:
                        total_enemy += 1
                        all_enemies.append(curr_cell.ship)
            # if curr_cell.shipyard:
            #     if is_enemy == True:
            #         if curr_cell.shipyard.player_id != me.id:
            #             total_enemy += 1
            #             all_enemies.append(curr_cell.ship)
            #     else:
            #         if curr_cell.shipyard.player_id == me.id:
            #             total_enemy += 1
            #             all_enemies.append(curr_cell.ship)

    enemies_around_point[(pt, size, is_enemy)] = (total_enemy, all_enemies)
    return total_enemy, all_enemies



def obtain_enemy_around_position(board, target_pt, size=1):
    for ii in list(range(-size, 0)) + [0] + list(range(1, size + 1)):
        curr_max_jj = abs(size - abs(ii))
        for jj in list(range(-curr_max_jj, 0)) + [0] + list(range(1, curr_max_jj + 1)):
            curr_pos = target_pt + Point(ii, jj)
            curr_pos = curr_pos % 21
            curr_cell = board.cells[curr_pos]
            if curr_cell.ship:
                if curr_cell.ship.player_id != me.id:
                    return False

            if curr_cell.shipyard:
                if curr_cell.shipyard.player_id != me.id:
                    return False

    return True


def obtain_my_ships_in_position(board, target_pt, size=2):
    num_my_ships = 0
    num_halite_list = []
    num_cargo_list = []
    num_halite_density = 0
    for ii in list(range(-size, 0)) + [0] + list(range(1, size + 1)):
        curr_max_jj = abs(size - abs(ii))
        for jj in list(range(-curr_max_jj, 0)) + [0] + list(range(1, curr_max_jj + 1)):
            if ii==0 and jj ==0:
                continue
            curr_pos = target_pt + Point(ii, jj)
            curr_pos = curr_pos % 21
            curr_cell = board.cells[curr_pos]
            if curr_cell.ship:
                if curr_cell.ship.player_id == me.id:
                    num_my_ships += 1
                    num_cargo_list.append(curr_cell.ship.halite)

            num_halite_list.append(curr_cell.halite)

    none_zero = [1 if w != 0 else 0 for w in num_halite_list]
    num_halite_density = sum(num_halite_list) / len(none_zero)

    return num_my_ships, sum(none_zero), sum(num_cargo_list), num_halite_density


def shipyard_positions(translate=False):
    if translate == False:
        return {shipyard.id: shipyard.position for shipyard in me.shipyards}
    else:
        return {shipyard.id: translate_pos(shipyard.position) for shipyard in me.shipyards}


def ship_positions(translate=False):
    if translate == False:
        return {ship.id: ship.position for ship in me.ships}
    else:
        return {ship.id: translate_pos(ship.position) for ship in me.ships}


#################################################################
# for each cell, calculate the halite density around it,
# weighted by 1 / distance_to_it
# used to assist shipyard/mining spot selection
#################################################################

def dist_weighted_mean_pool(kernel):
    num_rows, num_cols = kernel.shape
    center_row, center_col = num_rows // 2, num_cols // 2
    weights = {}
    for ii in range(num_rows):
        for jj in range(num_cols):
            weights[ii, jj] = abs(center_row - ii) + abs(center_col - jj)
    weights = 1 / (np.array(list(weights.values())).reshape(num_rows, num_cols) + 1e-10)
    weights[center_row, center_col] = 0
    values = np.array(kernel) * weights
    return np.mean(values)


def pool2d(A, kernel_size, stride, padding, pool_func):
    '''
    2D Pooling

    Parameters:
        A: input 2D array, dataframe
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    inds, cols = A.index, A.columns
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    res = np.array([pool_func(stride) for stride in A_w]).reshape(output_shape)
    res = pd.DataFrame(res,
                       index=inds[kernel_size[0] // 2:-kernel_size[0] // 2 + 1],
                       columns=cols[kernel_size[1] // 2:-kernel_size[1] // 2 + 1])
    return res


def get_dist_weighted_density(halites_snapshot, kernel_size):
    full_snapshot = pd.DataFrame(np.tile(halites_snapshot, (3, 3)), index=range(-21, 42), columns=range(-21, 42))
    orig = full_snapshot.loc[0 - kernel_size // 2: 20 + kernel_size // 2,
           0 - kernel_size // 2: 20 + kernel_size // 2]
    return pool2d(A=orig, kernel_size=kernel_size, stride=1, padding=0, pool_func=dist_weighted_mean_pool)


#################################################################
# check if is suitable to build shipyard
#################################################################

def adj1_get_convert_thred(position, remaining_steps):
    # convert if storage * (1 - 0.75 ** (400 - step)) > 500 - cargo:
    # 2.5 here is hyper param
    row, col = translate_pos(position)
    storage = slice(halite, row - 2, row + 2, col - 2, col + 2)
    return max(300,
               500 - np.sum(storage) / 30 * (1 - 0.75 ** (remaining_steps / 30)))


def adj2_good_region_to_convert(position):
    # if current region density is of top 5%
    row, col = translate_pos(position)

    # adj: v3_21_adj22_feedback3_full_adj9, bugfix, collision happen before convert, so need to make sure surranding cells are safe before convert
    filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if np.sum((slice(turn.flag_matrix, row - 1, row + 1, col - 1, col + 1) == 2) * filter) > 0:
        return False

    unique_density = np.unique(halite_density)
    k = -int(len(unique_density) * 0.05)

    top = unique_density[np.argpartition(unique_density, k)]
    if halite_density.loc[row, col] >= top[-int(len(unique_density) * 0.01)]:
        r = 3
    else:
        r = 2
    if np.sum(slice(turn.flag_matrix, row - r, row + r, col - r, col + r) == 3) > 0:
        return False

    return halite_density.loc[row, col] >= unique_density[np.argpartition(unique_density, k)[k]]

#################################################################
# regularly check potential good region to explore
#################################################################

def adj4_regular_check_on_high_halite_low_ship_density_region(topk):
    # import ipdb; ipdb.set_trace()
    unique_density = np.unique(halite_density)
    k = -int(len(unique_density) * 0.01)
    thred = unique_density[np.argpartition(unique_density, k)[k]]
    target_positions = np.argwhere(halite_density.values >= thred)
    # return [tuple(pos) for pos in target_positions]
    selected = set()
    q = deque(
        sorted([(halite_density.loc[pos[0], pos[1]], pos) for pos in target_positions],
               reverse=True, key=lambda x: x[0])
    )
    while q:
        _, pos = q.popleft()
        row, col = pos

        if np.sum(slice(turn.flag_matrix, row - 2, row + 2, col - 2, col + 2) == 4) >= 2:
            continue

        good = True
        for old_pos in selected:
            if circular_distance(old_pos[0], pos[0]) < 5 and circular_distance(old_pos[1], pos[1]) < 5:
                good = False
                break
        if good:
            selected.add(tuple(pos))

    closest_ships = adj8_find_closest_ships_to_regions(selected, topk)

    return closest_ships  # selected_region: closest_ship


def adj8_find_closest_ships_to_regions(regions, k):
    """
    select closest EMPTY ship to each region
    regions: [(y, x), ...] or [(row, col), ...]
    return {regions: [(halite_density.loc[region1], distance1, ship1),
                      (halite_density.loc[region2], distance2, ship2), ...]}
    """
    distances = defaultdict(list)
    for region_pos in regions:
        region_row, region_col = region_pos
        for ship in me.ships:
            if ship.halite != 0:
                continue
            ship_row, ship_col = translate_pos(ship.position)
            dist = circular_distance(region_row, ship_row) + circular_distance(region_col, ship_col)
            if dist == 0:
                hd_ratio = 10000
            else:
                hd_ratio = halite_density.loc[region_row, region_col] / dist
            distances[region_pos].append((hd_ratio, ship.id))

    for region_pos, dist in distances.items():
        dist.sort(key=lambda xx: xx[0], reverse=True)

    ratios = sorted(distances.items(), key=lambda x: x[1][0][0], reverse=True)

    pairs = {}
    visited_ships = set()
    for region_pos, ships in ratios:
        for _, ship_id in ships:
            if ship_id not in visited_ships:
                pairs[region_pos] = ship_id
                visited_ships.add(ship_id)
                break
        if len(pairs) >= k:
            break

    return pairs


def get_my_shipyard_average_density_pct():
    pcts = 0
    all_hdensities = halite_density.values.flatten()
    for shipyard in me.shipyards:
        row, col = translate_pos(shipyard.position)
        pcts += (all_hdensities < halite_density.loc[row, col]).sum() / all_hdensities.shape[0]
    if len(me.shipyards) == 0:
        pcts = 0
    else:
        pcts /= len(me.shipyards)
    return pcts


def get_my_shipyard_density_pct():
    pcts = {}
    all_hdensities = halite_density.values.flatten()
    for shipyard in me.shipyards:
        row, col = translate_pos(shipyard.position)
        pcts[shipyard.position] = (all_hdensities < halite_density.loc[row, col]).sum() / all_hdensities.shape[0]
    return pcts


def adj5_final_steps_constraint_low_avg_halites():
    # max ship last steps -15, seems not to work well
    # could try to exhaust
    pass


def adj6_spawn_more_if_have_shipyards_at_good_regions():
    # check more on spawn policy ([sljfwe])
    pass


def adj7_detect_aggressive_player_and_keep_away():
    pass


# increase number of ships limit?
# last steps, if my average cargo is low, don't make new ship any more
# before making new ship, count average ship / shipyard, and average cargo

# when select replacement shipyard when existing got colission, pay some attention to regional enemy, halite, etc. too
# (but maybe convert largest is still one of the most reasonable choice)

# i should make some aggressive strategies, to train and test my protective strategies

# think more about interactions with other ships (mine and enemy)
# add protective empty ship around shipyard
# what's best strategy with aggressive counterparty, escaping or aggressive?
# if i were to develop aggressive strategy, what should it be?
# markov
# adj7, v += more if a region has our shipyards (don't leave my shipyard as empty) v3/1768325.json

# ========================

'''
Initialization code can run when the file is loaded.  The first call to agent() is allowed 30 sec
The agent function is the last function in the file (does not matter its name)
agent must run in less than 6 sec
Syntax errors when running the file do not cause any message, but cause your agent to do nothing

'''
CONFIG_MAX_SHIPS = 40
CONFIG_MAX_SHIPYARDS = 8
MERGE_TIME = 395
FINAL_ATTACK_TIME = 385

# print('kaggle version',kaggle_environments.__version__)
#### Global variables
global attackers
attackers = {}
all_actions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
all_dirs = [Point(0, 1), Point(1, 0), Point(0, -1), Point(-1, 0)]
start = None
num_shipyard_targets = 4
size = None
# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
ship_target = {}  # where to go to collect
missions = {}  # {ship_id: position}
guarders = {}  # {shipyard_id: ship_id}
global guarder_activated
guarder_activated = False

global inplace_guarders
inplace_guarders = set()

global aggressive_player_detected
aggressive_player_detected = False

global lock_step
lock_step = 50  # first steps encourage more aggressive mining

# adj: v3_21_adj22_feedback3_adjtrunc, add global here
global prev_convert
prev_convert = -10000
global mission_done
mission_done = None
me = None
did_init = False
quiet = False
C = None


class Obj:
    pass


# will hold global data for this turn, updating as we set actions.
# E.g. number of ships, amount of halite
# taking into account the actions set so far.  Also max_ships, etc.
turn = Obj()
### Data
# turns_optimal[CHratio, round_trip_travel] for mining
# See notebook on optimal mining kaggle.com/solverworld/optimal-mining
turns_optimal = np.array(
    [[0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
     [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
     [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
     [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
     [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
     [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
     [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


#### Functions
def print_enemy_ships(board):
    print('\nEnemy Ships')
    for ship in board.ships.values():
        if ship.player_id != me.id:
            print('{:6}  {} halite {}'.format(ship.id, ship.position, ship.halite))


def print_actions(board):
    print('\nShip Actions')
    for ship in me.ships:
        print('{:6}  {}  {} halite {}'.format(ship.id, ship.position, ship.next_action, ship.halite))
    print('Shipyard Actions')
    for sy in me.shipyards:
        print('{:6}  {}  {}'.format(sy.id, sy.position, sy.next_action))


def print_none(*args):
    pass


def compute_max_ships(step):
    # This needs to be tuned, perhaps based on amount of halite left
    if step < 200:
        return CONFIG_MAX_SHIPS
    elif step < 300:
        return CONFIG_MAX_SHIPS - 2
    elif step < 350:
        return CONFIG_MAX_SHIPS - 4
    else:
        return CONFIG_MAX_SHIPS - 5


def set_turn_data(board):
    # initialize the global turn data for this turn
    turn.num_ships = len(me.ships)
    turn.max_ships = compute_max_ships(board.step)
    turn.total_halite = me.halite
    # this is matrix of halite in cells
    turn.halite_matrix = np.reshape(board.observation['halite'], (board.configuration.size, board.configuration.size))
    turn.num_shipyards = len(me.shipyards)
    # compute enemy presence and enemy halite matrices
    turn.EP, turn.EH, turn.ES = gen_enemy_halite_matrix(board)
    # filled in by shipid as a ship takes up a square
    turn.taken = {}
    turn.last_episode = (board.step == (board.configuration.episode_steps - 2))
    # import ipdb; ipdb.set_trace()
    turn.flag_matrix = np.zeros((board.configuration.size, board.configuration.size))
    # {1: my ship, 2: enemy ship, 3: my shipyard, 4: enemy shipyard}
    for id, ship in board.ships.items():
        row, col = translate_pos(ship.position)
        if ship.player_id == me.id:
            turn.flag_matrix[row, col] = 1
        else:
            turn.flag_matrix[row, col] = 2
    for id, sy in board.shipyards.items():
        row, col = translate_pos(sy.position)
        if sy.player_id == me.id:
            if turn.flag_matrix[row, col] == 1:
                turn.flag_matrix[row, col] = 5
            else:
                turn.flag_matrix[row, col] = 3
        else:
            if turn.flag_matrix[row, col] == 2:
                turn.flag_matrix[row, col] = 6
            else:
                turn.flag_matrix[row, col] = 4


def init(obs, config):
    # This is only called on first call to agent()
    # Do initalization things
    global size
    global print
    if hasattr(config, 'myval') and config.myval == 9 and not quiet:
        # we are called locally, so leave prints OK
        pass
    else:
        # we are called in competition, quiet output
        print = print_none
        pprint.pprint = print_none
    size = config.size


def limit(x, a, b):
    if x < a:
        return a
    if x > b:
        return b
    return x


def num_turns_to_mine(C, H, rt_travel):
    # How many turns should we plan on mining?
    # C=carried halite, H=halite in square, rt_travel=steps to square and back to shipyard
    if C == 0:
        ch = 0
    elif H == 0:
        ch = turns_optimal.shape[0]
    else:
        ch = int(math.log(C / H) * 2.5 + 5.5)
        ch = limit(ch, 0, turns_optimal.shape[0] - 1)
    rt_travel = int(limit(rt_travel, 0, turns_optimal.shape[1] - 1))
    return turns_optimal[ch, rt_travel]


def halite_per_turn(carrying, halite, travel, min_mine=1):
    # compute return from going to a cell containing halite, using optimal number of mining steps
    # returns halite mined per turn, optimal number of mining steps
    # Turns could be zero, meaning it should just return to a shipyard (subject to min_mine)
    turns = num_turns_to_mine(carrying, halite, travel)
    if turns < min_mine:
        turns = min_mine
    mined = carrying + (1 - .75 ** turns) * halite
    return mined / (travel + turns), turns


def move(pos, action):
    ret = None
    # return new Position from pos when action is applied
    if action == ShipAction.NORTH:
        ret = pos + Point(0, 1)
    if action == ShipAction.SOUTH:
        ret = pos + Point(0, -1)
    if action == ShipAction.EAST:
        ret = pos + Point(1, 0)
    if action == ShipAction.WEST:
        ret = pos + Point(-1, 0)
    if ret is None:
        ret = pos
    # print('move pos {} {} => {}'.format(pos,action,ret))
    return ret % size


def dirs_to(p1, p2, size=21):
    # Get the actions you should take to go from Point p1 to Point p2
    # using shortest direction by wraparound
    # Args: p1: from Point
    #      p2: to Point
    #      size:  size of board
    # returns: list of directions, tuple (deltaX,deltaY)
    # The list is of length 1 or 2 giving possible directions to go, e.g.
    # to go North-East, it would return [ShipAction.NORTH, ShipAction.EAST], because
    # you could use either of those first to go North-East.
    # [None] is returned if p1==p2 and there is no need to move at all
    deltaX, deltaY = p2 - p1
    if abs(deltaX) > size / 2:
        # we wrap around
        if deltaX < 0:
            deltaX += size
        elif deltaX > 0:
            deltaX -= size
    if abs(deltaY) > size / 2:
        # we wrap around
        if deltaY < 0:
            deltaY += size
        elif deltaY > 0:
            deltaY -= size
    # the delta is (deltaX,deltaY)
    ret = []
    if deltaX > 0:
        ret.append(ShipAction.EAST)
    if deltaX < 0:
        ret.append(ShipAction.WEST)
    if deltaY > 0:
        ret.append(ShipAction.NORTH)
    if deltaY < 0:
        ret.append(ShipAction.SOUTH)
    if len(ret) == 0:
        ret = [None]  # do not need to move at all
    return ret, (deltaX, deltaY)


#################################################################
# assign guarders to protect shipyards: method1, only consider closest enemy
# action value functions should call guarders back to shipyard when danger sensed
# need to trade-off between shipyard safety and mining efficiency
#################################################################


def shipyard_threatened(sy_position, rr):
    row, col = translate_pos(sy_position)
    return len(find_within([2], row, col, within=rr)) > 0


def find1(row, col, rr, use_min=False, below=10000):
    flag = slice(turn.flag_matrix, row - rr, row + rr, col - rr, col + rr)
    flag = has_my_ship(flag)
    invalid_points = np.argwhere(flag)
    invalid_points = invalid_points[np.abs((invalid_points - np.array(flag.shape) // 2)).sum(axis=1) > rr]
    flag[invalid_points[:, 0], invalid_points[:, 1]] = False
    cargo = slice(cargo_matrix, row - rr, row + rr, col - rr, col + rr)
    if np.sum(has_my_ship(flag)) == 0:
        return None

    if use_min:
        cargo = np.where(flag, cargo, 10000)
        candidates = np.argwhere(np.logical_and(cargo == np.nanmin(cargo), cargo <= below))
    else:
        cargo = np.where(flag, cargo, -10000)
        candidates = np.argwhere(cargo == np.nanmax(cargo))

    if not candidates.size:
        return None

    distances = candidates - np.array([rr, rr])
    distances = np.abs(distances)
    distances = np.nansum(distances, axis=1)
    candidate = candidates[np.argwhere(distances == min(distances))[0]]
    res = recover_pos(candidate, row, col, rr)
    return tuple(res[0])
    # return tuple(((np.argwhere(cargo == np.nanmax(cargo))[0] + np.array([row, col]) - rr) + size) % size)


def find2(row, col, rr):
    global cargo_matrix

    # guarder_todo: replace with within
    flag = slice(turn.flag_matrix, row - rr, row + rr, col - rr, col + rr) == 2
    invalid_points = np.argwhere(flag)
    invalid_points = invalid_points[np.abs((invalid_points - np.array(flag.shape) // 2)).sum(axis=1) > rr]
    flag[invalid_points[:, 0], invalid_points[:, 1]] = False
    if np.sum(flag) == 0:
        return 10000, None

    cargos = slice(cargo_matrix, row - rr, row + rr, col - rr, col + rr)
    valid_points = np.abs(np.argwhere(flag))
    distances = np.sum(np.abs(np.argwhere(flag) - np.array([rr, rr])), axis=1)
    nearest_point = valid_points[np.argmin(distances)]
    enemy_cargo = cargos[tuple(nearest_point)]
    return np.min(distances), enemy_cargo


# adj: v3_21_adj22_feedback3_full_adj15_adj16_v1: restructured this function, to update guarders every turn
def assign_guarders():
    # clean up dead shipyards
    # adj: v3_21_adj22_feedback3_adjtrunc_adj5
    global guarder_activated
    global aggressive_player_detected
    global inplace_guarders

    my_ships = set([ship.id for ship in me.ships])
    for ship_id in inplace_guarders:
        if ship_id not in my_ships:
            aggressive_player_detected = True

    pos2ship = {}
    for ship in me.ships:
        pos2ship[ship.position] = ship.id

    my_sypos = set([sy.position for sy in me.shipyards])
    to_del = []
    for sypos in guarders:
        # adj: v3_21_adj22_feedback3_full_adj16, this is a bug, but keeping it intentionally
        #      so condition after or always be true
        if sypos not in my_sypos or guarders[sypos] not in pos2ship:
            guarder_activated = True
            to_del.append(sypos)

    for sypos in to_del:
        del guarders[sypos]

    if len(me.ships) == 1 and len(me.shipyards) == 1 and (me.halite < 500
                                                          or board.step >= 350
                                                          or aggressive_player_detected):
        return

    rr = 7
    for sy in me.shipyards:
        if sy.next_action == ShipyardAction.SPAWN:
            continue
        pos = sy.position
        # adj: v3_21_adj22_feedback3_full_adj7, bugfix, translate_pos in this section
        row, col = translate_pos(pos)
        # search for ship within rr range, with highest cargo
        closest_enemy, enemy_cargo = find2(row, col, rr)
        if closest_enemy < 10000:
            guarder_activated = True
        pos_guarder = find1(row, col, rr=min(rr, closest_enemy - 1), use_min=False)
        if not pos_guarder:
            pos_guarder = find1(row, col, rr=min(rr, closest_enemy), use_min=True, below=enemy_cargo)
        if pos_guarder:
            y, x = 20 - pos_guarder[0], pos_guarder[1]
            guarders[pos] = pos2ship[(x, y)]

    inplace_guarders = set()
    for ship in me.ships:
        if ship.position in guarders:
            inplace_guarders.add(ship.id)

    return

def obtain_nearest_sy_distance_from_list(target_pt, sy_list):
    mn = 100
    best_pos = None
    avg_sy_dist_list = []
    for sy in sy_list:
        d = dist(target_pt, sy)
        avg_sy_dist_list.append(d)
        if d < mn:
            mn = d
            best_pos = sy

    if len(avg_sy_dist_list) == 0:
        avg_sy_dist_list = 0
    else:
        avg_sy_dist_list = np.mean(avg_sy_dist_list)

    return mn, best_pos, avg_sy_dist_list

def obtain_halite_density_within_my_area():
    my_protect_pts = []

    if len(board.current_player.shipyards) == 0:
        protect_zone_r1 = 3
    else:
        protect_zone_r1 = 3 + 4 * np.log2(len(board.current_player.shipyards))

    my_shipyards_list = [sy.position for sy in board.current_player.shipyards]
    for pt, c in board.cells.items():
        min_dist_my_sy, min_my_sy, avg_sy_dist = obtain_nearest_sy_distance_from_list(pt, my_shipyards_list)
        if min_my_sy != None and avg_sy_dist <= protect_zone_r1:  # if no shipyard, then no protect
            my_protect_pts.append(pt)

    if len(my_protect_pts) == 0:
        return 0
    else:
        my_protect_pts_halites = [board.cells[curr_pt].halite for curr_pt in my_protect_pts if board.cells[curr_pt].halite != 0]
        return np.mean(my_protect_pts_halites)

    return 0

def shipyard_actions():
    # spawn a ship as long as there is no ship already moved to this shipyard

    def spawn():
        if (
                len(me.shipyards) <= 1
                and len(me.ships) >= 20
                and shipyard_threatened(me.shipyards[0].position, rr=5)
                and guarders.get(me.shipyards[0].position)
                and turn.total_halite < 1000
                and max([ship.halite for ship in me.ships]) < 500):
            return
        if (board.step < 350 and turn.total_halite >= 500 and sy.position not in turn.taken) or (len(me.ships) == 0):
            # spawn one
            sy.next_action = ShipyardAction.SPAWN
            turn.taken[sy.position] = 1
            turn.num_ships += 1
            turn.total_halite -= 500
            is_spawn_flag = True
            return

    my_ships = [ship.halite for ship in me.ships]
    average_cargo = np.average(my_ships) if my_ships else float('inf')
    average_halite = np.average(halite[np.where(halite != 0)])

    # sort my shipyards by f(halite_density, my_ships_counts nearby), from high to low,
    # TODO: leaving out my_ships_counts for now, will revisit necessity later
    my_shipyards = sorted(me.shipyards, key=lambda sy: halite_density.loc[translate_pos(sy.position)], reverse=True)
    # assign_guarders()

    my_halite_density = obtain_halite_density_within_my_area()
    # print_(board.step, my_halite_density)
    # my_shipyard_area_density_list = []
    # for sy in my_shipyards:
    #     num_my_ship, mean_halite, num_cargo, my_halite_density = obtain_my_ships_in_position(board, sy.position,
    #                                                                                          size=2)
    #     my_shipyard_area_density_list.append(my_halite_density)
    #
    # if len(my_shipyard_area_density_list) != 0:
    #     my_halite_density = max(my_shipyard_area_density_list)
    # else:
    #     my_halite_density = 500

    is_spawn_flag = False
    for sy in my_shipyards:
        if sy.position in guarders and not shipyard_threatened(sy.position, rr=5) and turn.total_halite >= 500:
            del guarders[sy.position]

        if not turn.last_episode and len(me.ships) == 0:
            spawn()
            continue

        if board.step > 120 and is_spawn_flag == True:            # do not spawn too many ships in mid-late game
            continue

        if turn.num_ships < turn.max_ships:
            if board.step <= 80:
                spawn()
            elif board.step <= 120 and my_halite_density > 40:
                spawn()
            elif board.step <= 150 and my_halite_density > 50:
                spawn()
            elif board.step <= 200 and my_halite_density > 60:
                spawn()
            elif board.step <= 250:
                if my_halite_density > 80:
                    spawn()
            elif board.step <= 300:
                if my_halite_density > 90:
                    spawn()
            elif board.step < 350:
                if my_halite_density > 100:
                    spawn()
            else:
                pass


def obtain_nearest_myshipyard(board, pts1, size=3, sy_size=2):
    # size: the size to check nearby enemy
    # sy_size: the size to check nearby shipyard
    is_nurtuing_dict = {}
    for pt in pts1:
        is_close_to_shipyard = False
        is_enemy_nearby = False
        for ii in list(range(-size, 0)) + [0] + list(range(1, size + 1)):
            curr_max_jj = abs(size - abs(ii))
            for jj in list(range(-curr_max_jj, 0)) + [0] + list(range(1, curr_max_jj + 1)):
                curr_pos = pt + Point(ii, jj)
                curr_pos = curr_pos % 21
                curr_cell = board.cells[curr_pos]
                if curr_cell.ship:
                    if curr_cell.ship.player_id != me.id:
                        is_enemy_nearby = True
                        break
                if curr_cell.shipyard:
                    if curr_cell.shipyard.player_id != me.id:
                        is_enemy_nearby = True
                        break
                    if curr_cell.shipyard.player_id == me.id and abs(ii)+abs(jj)<=sy_size:
                        is_close_to_shipyard = True

            if is_enemy_nearby == True:
                break

        if is_enemy_nearby == False and is_close_to_shipyard == True and board.cells[pt].halite < 150:
            is_nurtuing_dict[pt] = True
        else:
            is_nurtuing_dict[pt] = False

    return is_nurtuing_dict

def check_target_position_safety(board, pts1, size=2):
    is_place_safe_dict = {}
    for pt in pts1:
        is_safe = True
        for ii in list(range(-size, 0)) + [0] + list(range(1, size + 1)):
            curr_max_jj = abs(size - abs(ii))
            for jj in list(range(-curr_max_jj, 0)) + [0] + list(range(1, curr_max_jj + 1)):
                curr_pos = pt + Point(ii, jj)
                curr_pos = curr_pos % 21
                curr_cell = board.cells[curr_pos]
                if curr_cell.ship:
                    if curr_cell.ship.player_id != me.id:
                        is_safe = False
                if curr_cell.shipyard:
                    if curr_cell.shipyard.player_id != me.id:
                        is_safe = False
                if is_safe == False:
                    break
            if is_safe == False:
                break

        if is_safe == True and board.cells[pt].halite >= 50:
            is_place_safe_dict[pt]  = True
        else:
            is_place_safe_dict[pt] = False

    return is_place_safe_dict

def get_ship2shipyard_distances():
    remaining_steps = 398 - board.step

    closest_shipyards = {}
    for player in board.players.values():
        for ship in player.ships:
            closest_shipyards[ship] = [10000, None]
            for shipyard in player.shipyards:
                ship2shipyard_dist = dist(ship.position, shipyard.position)
                if ship2shipyard_dist > remaining_steps:
                    continue
                if ship2shipyard_dist < closest_shipyards[ship][0]:
                    closest_shipyards[ship] = [ship2shipyard_dist, shipyard]

    closest_ships = {}
    for ship, (ship2shipyard_dist, shipyard) in closest_shipyards.items():
        if ship2shipyard_dist > remaining_steps:
            continue
        if shipyard not in closest_ships:
            closest_ships[shipyard] = []
        closest_ships[shipyard].append(ship)

    return closest_shipyards, closest_ships


from scipy.stats import rankdata

def final_attacks():
    """
    0. gather all my empty ships, assign prioritized tasks (guarders)
    1. get all players halite + cargo (within remaining steps distance to shipyard)
    2. get all players rank
    3. get my prev and next rank
    4. get my prev and next difference
    5. get the closer difference one
    6. attack its shipyard
    """
    if board.step > 398:
        return

    remaining_steps = 398 - board.step
    closest_shipyards, closest_ships = get_ship2shipyard_distances()
    guarder_ships = set(guarders.values())
    # simplified for now, just adding all cargos to halites
    # to do: only add cargos in ships that are close enough to sent back to shipyard
    player_halites = [player.halite for player in board.players.values()]
    player_cargos = []
    for player_id, player in enumerate(board.players.values()):
        cargo = 0
        for ship in player.ships:
            if closest_shipyards[ship][0] <= remaining_steps:
                cargo += ship.halite
            elif ship.halite > 500:
                cargo += ship.halite - 500
        player_cargos.append(cargo)

    player_scores = [halite + cargo for halite, cargo in zip(player_halites, player_cargos)]
    ranks = 4 - rankdata(player_halites, method='ordinal')
    my_rank = ranks[me.id]
    if my_rank == 0:
        to_attack = [ii for ii in range(4) if ranks[ii] == 1][0]
    elif my_rank == 3:
        to_attack = [ii for ii in range(4) if ranks[ii] == 2][0]
    else:
        prev_player = np.argwhere(ranks == my_rank - 1)[0,0]
        next_player = np.argwhere(ranks == my_rank + 1)[0,0]
        prev_score, my_score, next_score = (player_scores[prev_player],
                                            player_scores[me.id],
                                            player_scores[next_player])
        if next_score >= player_halites[me.id] + 0.5 * player_cargos[me.id]:
            to_attack = next_player
        else:
            to_attack = prev_player

    empty_ships = [ship for ship in me.ships if ship.halite == 0 and ship.id not in guarder_ships and ship.id not in attackers]

    target_shipyards = []
    for shipyard in board.players[to_attack].shipyards:
        if not closest_ships.get(shipyard, []):
            continue
        my_ship_dists = [dist(ship.position, shipyard.position) for ship in empty_ships]
        my_ship_dists = [dst for dst in my_ship_dists if dst < remaining_steps]
        if not my_ship_dists:
            continue
        my_furthest_ship_dist = max(my_ship_dists)
        my_closest_ship_dist = min(my_ship_dists)
        cargo_min = np.sum(ship.halite for ship in closest_ships[shipyard] if closest_shipyards[ship][0] >= my_furthest_ship_dist)
        cargo_max = np.sum(
            ship.halite for ship in closest_ships[shipyard] if closest_shipyards[ship][0] >= my_closest_ship_dist)
        cargo_mid = (cargo_min + cargo_max) / 2.
        target_shipyards.append((cargo_mid, shipyard))

    target_shipyards.sort(key=lambda xx: xx[0], reverse=True)

    if not target_shipyards or not target_shipyards[0][1]:
        return

    # method1: attacking only one shipyard
    target = target_shipyards[0][1]
    empty_ships = [ship for ship in empty_ships if dist(ship.position, target.position) < remaining_steps]
    for ship in empty_ships:
        ship_target[ship.id] = target.position
        attackers[ship.id] = target.position

    # TODO
    # method2: attack as many shpyards as possible

    print_(board.step, 'attackers', '*' * 30, {ship: (board.ships[ship].position, target) for (ship, target) in attackers.items()})
    return


target_turns = {}

def assign_targets(board, ships):
    global attackers
    global PROTECT_MODE
    # Assign the ships to a cell containing halite optimally
    # set ship_target[ship_id] to a Position
    # We assume that we mine halite containing cells optimally or return to deposit
    # directly if that is optimal, based on maximizing halite per step.
    # Make a list of pts containing cells we care about, this will be our columns of matrix C
    # the rows are for each ship in collect
    # computes global dict ship_tagert with shipid->Position for its target
    # global ship targets should already exist
    old_target = copy.copy(ship_target)
    ship_target.clear()
    if len(ships) == 0:
        return

    # halite_min = 50
    global_mean = np.mean(turn.halite_matrix[turn.halite_matrix != 0])
    halite_min = min(global_mean * 0.75, 50)

    pts1 = []
    pts2 = []

    good_positions = adj4_regular_check_on_high_halite_low_ship_density_region(topk=2)
    tot_empty_ships = sum(1 for ship in me.ships if ship.halite == 0)

    assign_guarders()

    # delete dead ship from memo
    my_ship_ids = {ship.id: ship for ship in me.ships}
    to_del_ids = [ship_id for ship_id in missions if ship_id not in my_ship_ids]
    for ship_id in to_del_ids:
        del missions[ship_id]

    # delete ship that completed mission from memo
    best_region = None
    to_del_ids = []
    for ship_id in missions:
        ship_row, ship_col = translate_pos(my_ship_ids[ship_id].position)
        target_row, target_col = missions[ship_id]
        curr_hd = halite_density.loc[target_row, target_col]
        if best_region is None:
            best_region = (curr_hd, ship_id)
            mission_done = ship_id
        else:
            if curr_hd > best_region[0]:
                best_region = (curr_hd, ship_id)
                mission_done = ship_id

        if circular_distance(ship_col, target_col) + circular_distance(ship_row, target_row) < 5:
            to_del_ids.append(ship_id)

    for ship_id in to_del_ids:
        del missions[ship_id]

    # my_sy_avg_density = get_my_shipyard_average_density_pct()
    halite_density_pct = get_my_shipyard_density_pct()
    if len(me.shipyards) == 0:
        my_sy_avg_density = 0
    else:
        my_sy_avg_density = sum(halite_density_pct.values()) / len(me.shipyards)

    covered = {}

    if board.step < 120:
        density_thred = 0.7  # adj: v3_21_adj22_feedback3_full_adj8, 0.5->0.7, <200 -> <150
    else:
        density_thred = 0.9

    # # adj: v3_21_adj22_feedback2, remove dups
    for pt, c in board.cells.items():
        assert isinstance(pt, Point)
        if c.halite > halite_min:
            pts1.append(pt)
    # #Now add duplicates for each shipyard - this is directly going to deposit
    # for sy in me.shipyards:
    #   for i in range(num_shipyard_targets):
    #     pts2.append(sy.position)
    for sy in me.shipyards:
        pts2.append(sy.position)

    # nearest my shipyard, and no enemy close-by
    is_nurtuing_dict = obtain_nearest_myshipyard(board, pts1)

    is_place_safe = check_target_position_safety(board, pts1)

    # this will be the value of assigning C[ship,pt]
    C = np.zeros((len(ships), len(pts1) + len(pts2)))
    # this will be the optimal mining steps we calculated
    for i, ship in enumerate(ships):
        d3, shipyard_position3 = nearest_shipyard(ship.position)
        if shipyard_position3 is None:
            d3 = 1

        if ship.halite > 0 and (d3 >= 398 - board.step - 1 or PROTECT_MODE):
            print_('+' * 30, f'step={board.step}', f'Forcing ship return: {ship.id}, {ship.position} -> {shipyard_position3}')
            ship_target[ship.id] = shipyard_position3
            continue

        for j, pt in enumerate(pts1 + pts2):
            # two distances: from ship to halite, from halite to nearest shipyard
            d1 = dist(ship.position, pt)
            d2, shipyard_position = nearest_shipyard(pt)
            pt_position = translate_pos(pt)
            if shipyard_position is None:
                # don't know where to go if no shipyard
                d2 = 1
            # value of target is based on the amount of halite per turn we can do
            my_halite = ship.halite
            if j < len(pts1):
                # in the mining section

                v, mining = halite_per_turn(my_halite, board.cells[pt].halite, d1 + d2)

                if d2 * 1.5 >= abs(398 - board.step) or (d3 + 5 >= 398 - board.step):
                    print('ending' * 3, '#' * 30)
                    v -= 100

                if board.step > 150 and board.step < 350 and is_nurtuing_dict[pt]:
                    # print('$$$$$ wait for a while')
                    v -= 50

                # continue to mine, if the second place if safe, and it is not far away from curr_position, and not far away from the shipyard
                if (ship.position != pt and
                    board.cells[pt].halite > board.cells[ship.position].halite and
                    ship.halite != 0 and
                    is_place_safe[pt] == True and
                    d1<=2 and
                    d2<=d3+2 and
                    board.cells[ship.position].halite < 50
                ):
                    v += 5

                if len(me.shipyards) == 0:
                    distsum = 0
                else:
                    distsum = sum([dist(ship.position, sy.position) for sy in me.shipyards])
                v -= distsum

                # adj: v3_21_adj22_feedback_full_adj9, restrict all cases when d2 > 10
                # adj: v3_21_adj22_feedback_full_adj10, restrict all cases when d2 > 50
                if d2 > 10:
                    v -= 300

                if ship.halite > 0 and find_within(targets=[6], row=pt_position[0], col=pt_position[1], within=1):
                    v -= 200

                # drive attackers away
                if (PROTECT_MODE
                      and turn.flag_matrix[pt_position[0], pt_position[1]] == 2
                      and cargo_matrix[pt_position[0], pt_position[1]] == 0
                      and d2 < 5
                      # and dist(ship.position, shipyard_position) < d2
                      and in_middle(mid_point=ship.position,
                                    start_point=shipyard_position,
                                    end_point=pt)
                      and ship.halite == 0
                      and ship.id not in set(guarders.values())
                      ):
                    print_('+' * 30, f'Driving enemy {ship.id}, {ship.position} -> {pt}')
                    cnt_empty_enemy = get_enemies_around_point(board, pt, 3, is_enemy=True)[1]
                    cnt_empty_enemy = len([ship for ship in cnt_empty_enemy if ship.halite == 0])
                    v += 10000 + cnt_empty_enemy * 100 - d1 * 20 - d2 * 20
                    attackers[ship.id] = pt
                # adj: v3_21_adj22_feedback3_full_adj7: bugfix, remove continue, need to add v to C
                # persist previous mining mission
                # adj: v3_21_adj22_feedback3_full_adj17: bugfix, pt.y,pt.x -> pt_position
                elif (board.step < 100
                        and pt.y == ship.position.y and pt.x == ship.position.x
                        and pt in target_turns
                        and target_turns[pt][1] == ship.id
                        and target_turns[pt][0] >= 1
                        and np.sum(slice(turn.flag_matrix, pt_position[0] - 2, pt_position[0] + 2, pt_position[1] - 2,
                                         pt_position[1] + 2) == 2) == 0
                ):
                    v += 100

                # persist previous expansion mission
                elif (my_sy_avg_density < density_thred
                      and board.step - prev_convert > 15
                      and ship.halite == 0
                      and len(missions) < min(3, tot_empty_ships * 0.25)
                      and pt_position in good_positions
                      and ship.id == good_positions[pt_position]
                ):
                    if ship.id not in missions or pt_position == missions[ship.id]:
                        if ship.id not in missions:
                            missions[ship.id] = pt_position
                        v += 10

                elif (
                        ship.halite == 0
                        and get_cargo_ratio(pt_position[0], pt_position[1], rr=2) > 1.5
                        and get_ship_ratio(pt_position[0], pt_position[1], rr=2) < 1
                ):
                    v += 10

                elif (
                        ship.halite == 0
                        and d2 < 2  # adj: v3_21_adj22_feedback2, < 5 -> < 2
                        and turn.flag_matrix[pt_position[0], pt_position[1]] == 2
                        and cargo_matrix[pt_position[0], pt_position[1]] == 0
                        and (pt not in covered or dist(covered[pt][0], pt) > dist(ship.position, pt))
                ):
                    v += 1300
                    if pt in covered:
                        prev_i = covered[pt][1]
                        C[prev_i, j] -= 1300
                    covered[pt] = [ship.position, i]

                # restrict to close-to-shipyard region # adj: v3_21_adj22_feedback3: d2 > 10 -> d2 > 5
                # adj: v3_21_adj22_feedback3_adjtrunc_adj6: -> strictly restrict to 5, without tolerance
                ## hold, adj: v3_21_adj22_feedback3_full_adj9, bugfix, ship_ratio > 1 -> < 1
                # adj: v3_21_adj22_feedback3_adj11: >5 -> >10
                elif d2 > 10 and (get_ship_ratio(pt_position[0], pt_position[1],
                                                 7) < 1  # adj: v3_21_adj22_feedback3_full_adj7: 3->7,
                                  or get_min_enemy_cargo(pt_position[0], pt_position[1], 2) < ship.halite):
                    # elif d2 > 5:
                    v -= 10

                    # TODO: [sljfwe] extra check on shipyards in that region
                # mining is no longer 0, due to min_mine (default)
            else:
                # in the direct to shipyard section
                if d1 > 0:
                    v = my_halite / d1
                    if d1 > 0:
                        v = my_halite / d1
                        if ((
                                guarder_activated
                                or (board.step > lock_step and halite_density_pct[pt] > 0.8)
                        )
                                and (
                                        len(find_within([2], pt_position[0], pt_position[1], within=d1 + 1)) > 0
                                )
                        ):
                            if guarders.get(pt) == ship.id:
                                v += 20000
                else:
                    # we are at a shipyard
                    v = 0

                    # adj: v3_21_adj22_feedback3_full_adj13: shift if block, seems to be a bug
                    if ((
                            guarder_activated
                            or (board.step > lock_step and halite_density_pct[pt] > 0.8)
                    )
                            and (
                                    # np.sum(slice(turn.flag_matrix, pt_position[0] - 7, pt_position[0] + 7,
                                    #              pt_position[1] - 7, pt_position[1] + 7) == 2) > 0
                                    len(
                                        find_within([2], pt_position[0], pt_position[1], within=5)
                                    ) > 0
                            )
                    ):
                        if guarders.get(pt) == ship.id:
                            v += 20000
                    ## adj: v3_21_adj22_feedback2, changing 10 to 50 and add d2 < 5, add elif
                    # adj: v3_21_adj22_feedback3_full_adj8: <5 -> <3
                    # # adj: v3_21_adj22_feedback3_full_adj14: tryy comment this out
                    # adj: v3_21_adj22_feedback3_full_adj18: try comment this back
                    elif 10 < board.step < lock_step and d2 < 3:
                        v -= 3
            if board.cells[pt].ship and board.cells[pt].ship.player_id != me.id:
                # if someone else on the cell, see how much halite they have
                # enemy ship
                enemy_halite = board.cells[pt].ship.halite
                if enemy_halite <= my_halite:
                    v -= 1000  # don't want to go there
                else:
                    if d1 < 2:
                        # attack or scare off if reasonably quick to get there
                        v += enemy_halite / (d1 + 1)  # want to attack them or scare them off
            # print('shipid {} col {} is {} with {:8.1f} score {:8.2f}'.format(ship.id,j, pt,board.cells[pt].halite,v))
            C[i, j] = v
    print('C is {}'.format(C.shape))
    # Compute the optimal assignment
    row, col = scipy.optimize.linear_sum_assignment(C, maximize=True)
    # so ship row[i] is assigned to target col[j]
    # print('got row {} col {}'.format(row,col))
    # print(C[row[0],col[0]])
    pts = pts1 + pts2
    for r, c in zip(row, col):
        if ship_target.get(ships[r].id):
            continue
        ship_target[ships[r].id] = pts[c]

    if board.step > FINAL_ATTACK_TIME:
        final_attacks()
    else:
        # attackers = {}
        if me.halite > 0:
            if len(me.ships) > 20:
              thred = 7
            elif len(me.ships) > 10:
              thred = 4
            else:
              thred = 0

            # thred = 7

            # attack closeby enemy shipyard
            enemy_sy_positions = []
            for enemy_sy in board.shipyards.values():
                enemy_id = enemy_sy.player_id
                if enemy_id == me.id:
                    continue
                enemy_pos = enemy_sy.position
                nearest_dist, my_pos = nearest_shipyard(enemy_pos)
                if nearest_dist > thred:
                    continue

                _, enemies = get_enemies_around_point(board, enemy_sy.position, size=nearest_dist, is_enemy=True)
                _, mine = get_enemies_around_point(board, my_pos, size=nearest_dist, is_enemy=False)
                enemies_cargos = np.mean([ship.halite for ship in enemies]) if enemies else 10000
                my_cargos = np.mean([ship.halite for ship in mine]) if mine else 10000

                if len(enemies) > len(mine) and enemies_cargos < my_cargos:
                    continue

                enemy_sy_positions.append(enemy_pos)


            if enemy_sy_positions:
                # print_('attacker triggered', '#' * 30)
                target = enemy_sy_positions[0]
                # dist_to_my_sy = nearest_shipyard(target)[0]
                my_empty_ships = sorted([ship for ship in me.ships if ship.halite == 0],
                                        key=lambda ship: nearest_shipyard(ship.position)[0])[len(me.shipyards):]
                my_empty_ships = [ship for ship in my_empty_ships if
                                  ship.id not in set(guarders.values())]  # and ship.id not in missions]
                for ship in my_empty_ships:
                    if ship.id not in attackers and (
                            me.halite > 1000 or dist(ship.position, target) < 5 and turn.flag_matrix[
                        translate_pos(target)] == 4):
                        ship_target[ship.id] = target
                        attackers[ship.id] = target


    # print out results
    # cview = view_C(C[3], pts)
    print('\nShip Targets')
    print('Ship      position          target')
    for id, t in ship_target.items():
        st = ''
        ta = ''

        if t not in target_turns or target_turns[t][0] <= 0:
            if board.step < 50:
                target_turns[t] = [5, id]
            else:
                target_turns[t] = [3, id]

        if board.ships[id].position == t:
            st = 'MINE'
            target_turns[t][0] -= 1
        elif len(me.shipyards) > 0 and t == me.shipyards[0].position:
            st = 'SHIPYARD'
        if id not in old_target or old_target[id] != ship_target[id]:
            ta = ' NEWTARGET'
        print('{0:6}  at ({1[0]:2},{1[1]:2})  assigned ({2[0]:2},{2[1]:2}) h {3:3} {4:10} {5:10}'.format(
            id, board.ships[id].position, t, board.cells[t].halite, st, ta))

    return


def make_avoidance_matrix(myship_halite):
    # make a matrix of True where we want to avoid, uses
    # turn.EP=enemy position matrix
    # turn.EH=enemy halite matrix
    # turn.ES=enemy shipyard matrix
    global avoid_matrixes
    if myship_halite in avoid_matrixes:
        return avoid_matrixes[myship_halite]

    filter = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if board.step < 50 and myship_halite == 0:
        bad_ship = np.logical_and(turn.EH < myship_halite, turn.EP)  # at the beginning of the game, the enemy is also scared. So, let me take the first bit then run away
        avoid = scipy.ndimage.convolve(bad_ship.astype(int), filter, mode='wrap', cval=0.0) >= 3

        bad_ship1 = np.zeros((size, size))
        for ship in board.ships.values():
            if ship.player_id in noavd_players:
                bad_ship1[ship.position] = 1

        avoid1 = scipy.ndimage.convolve(bad_ship1.astype(int), filter, mode='wrap', cval=0.0) >= 1

        avoid = np.logical_or(avoid, avoid1)

    else:
        bad_ship = np.logical_and(turn.EH <= myship_halite, turn.EP).astype(int)
        avoid = scipy.ndimage.convolve(bad_ship.astype(int), filter, mode='wrap', cval=0.0) >= 1
    avoid = np.logical_or(avoid, turn.ES)
    avoid_matrixes[myship_halite] = avoid
    return avoid


def make_attack_matrix(myship_halite):
    # make a matrix of True where we would want to move to attack an enemy ship
    # for now, we just move to where the ship is.
    # turn.EP=enemy position matrix
    # turn.EH=enemy halite matrix
    attack = np.logical_and(turn.EH > myship_halite, turn.EP)
    # print('attack',attack)
    return attack


def get_max_halite_ship(board, avoid_danger=True):
    # Return my Ship carrying max halite, or None if no ships
    # NOTE: creating avoid matrix again!
    mx = -1
    the_ship = None
    for ship in me.ships:
        x = ship.position.x
        y = ship.position.y
        avoid = make_avoidance_matrix(ship.halite)
        if ship.halite > mx and (not avoid_danger or not avoid[y, x]):
            mx = ship.halite
            the_ship = ship
    return the_ship


def remove_dups(p):
    # remove duplicates from a list without changing order
    # Not efficient for long lists
    ret = []
    for x in p:
        if x not in ret:
            ret.append(x)
    return ret


def matrix_lookup(matrix, pos):
    return matrix[pos.y, pos.x]


#################################################################
# ship converts (shipyard building)
#################################################################

def should_convert(ship):
    row, col = translate_pos(ship.position)
    is_dangerous = (
            (len(find_within([2], row, col, within=1)) >= len(find_within([1], row, col, within=1)))
                    and (len(find_within([2], row, col, within=2)) > len(find_within([1], row, col, within=1)))
    )
    if ship.halite == 0 and len(me.ships) == 1 and is_dangerous:
        return False
    if ship.halite + me.halite < 1000 and is_dangerous:
        return False
    return True

def get_max_mixed_score(avoid_danger=True):
    # Return my Ship carrying max halite, or None if no ships
    # NOTE: creating avoid matrix again!
    mx = -1
    the_ship = None
    for ship in me.ships:
        x = ship.position.x
        y = ship.position.y
        avoid = make_avoidance_matrix(ship.halite)
        num_friends, around_friendships = get_enemies_around_point(board, ship.position, size=3, is_enemy=False)
        num_enemies, around_enemiesships = get_enemies_around_point(board, ship.position, size=3, is_enemy=True)
        num_friends, mean_halite, my_cargos, my_halite_density = obtain_my_ships_in_position(board, ship.position,
                                                                                             size=2)
        curr = (
            np.sum(ship.halite for ship in around_friendships) if around_friendships else 0
            + 10 * num_friends
            - 10 * num_enemies
        )
        if (
                curr > mx
                and (not avoid_danger or not avoid[y, x])
        ):
            mx = curr
            the_ship = ship
    return the_ship

last_convert_timer = 0
attacker_ship_ids = set()

def ship_converts(board):
    global last_convert_timer

    def converts(ship, turn):
        print('converting:\t', ship.position)
        ship.next_action = ShipAction.CONVERT
        turn.taken[ship.position] = 1
        turn.num_shipyards += 1
        turn.total_halite -= 500

    # if no shipyard, convert the ship carrying max halite unless it is in danger
    expansion_done = False

    # adj: v3_21_adj22_feedback2: comment out init search to speed up initial accumulate
    # //*
    # adj3_init_convert
    if board.step == 0 and get_my_shipyard_average_density_pct() < 0.5:  # at initial step take max density within 1 ~ 2 step
        ship = me.ships[0]
        init_row, init_col = translate_pos(ship.position)
        max_halite = halite_density.loc[init_row, init_col]
        best_dir = (0, 0)
        for row_dir, col_dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            row, col = init_row + row_dir, init_col + col_dir
            if halite_density.loc[row, col] - max_halite > max_halite * 1:
                max_halite = halite_density.loc[row, col]
                best_dir = (row_dir, col_dir)

        if best_dir == (0, 0):
            ship.next_action = ShipAction.CONVERT
            if board.step < MERGE_TIME:
                turn.taken[ship.position] = 1
            turn.num_shipyards += 1
            turn.total_halite -= 500
        elif best_dir == (0, 1):
            ship.next_action = ShipAction.EAST
        elif best_dir == (1, 0):
            ship.next_action = ShipAction.SOUTH
        elif best_dir == (0, -1):
            ship.next_action = ShipAction.WEST
        elif best_dir == (-1, 0):
            ship.next_action = ShipAction.NORTH


    elif turn.num_shipyards == 0 and not turn.last_episode:
        # *// continue with if turn.num_shipyards==0 and not turn.last_episode:
        # if turn.num_shipyards==0 and not turn.last_episode:
        print_('<<<', board.step, 'triggered-sy-init-1')
        mx = get_max_mixed_score(board)
        if mx is not None:
            print_('triggered-sy-init-2')
            if mx.halite + turn.total_halite > 500 and should_convert(mx):
                mx.next_action = ShipAction.CONVERT
                if board.step < MERGE_TIME:
                    turn.taken[mx.position] = 1
                turn.num_shipyards += 1
                turn.total_halite -= 500
                print_(board.step, 'triggered-sy-init-2 >>>')

    # Now check the rest to see if they should convert
    for ship in me.ships:
        if ship.next_action:
            continue
        # CHECK if in danger without escape, convert if h>500
        avoid = make_avoidance_matrix(ship.halite)
        z = [matrix_lookup(avoid, move(ship.position, a)) for a in all_actions]
        if np.all(z) and ship.halite > 500:
            converts(ship, turn)
            print('ship id {} no escape converting'.format(ship.id))

        # CHECK if last step and > 500 halite, convert
        if turn.last_episode and ship.halite > 500:
            converts(ship, turn)

    board_mean_halite = np.mean(turn.halite_matrix[turn.halite_matrix != 0])
    # Generate shipyard from the best ship
    if (
            (turn.num_shipyards <= 5 and turn.num_ships >= turn.num_shipyards * 10
                and board.step < 300 and board.step - last_convert_timer >= 20 and board_mean_halite >= 40)
            or
            (
                # board.step <= 20 and turn.num_ships >= turn.num_shipyards * 9
                False
            )
    ):
        # check whether a ship dist to other shipyard is within an threshold
        ship_convert_score = []
        for ship in me.ships:
            # at the begining of the game, do not want shipyard built close to the center
            if board.step < 100 and abs(ship.position[0] - 10) <= 2 and abs(ship.position[1] - 10) <= 2:
                continue
            if ship.id in attacker_ship_ids:  # <-- ignore attackers
                continue
            is_distance_OK = True  # <-- do not want to build too close/far-away
            distance_list = []
            for shipyard in me.shipyards:
                curr_dist = dist(ship.position, shipyard.position)
                if curr_dist < 6:
                    is_distance_OK = False
                    break
                else:
                    distance_list.append(curr_dist)

            if is_distance_OK and len(distance_list) != 0:
                if min(distance_list) > 7:
                    is_distance_OK = False

            num_dist_le_7 = len([w for w in distance_list if w <= 7])  # want to the shipyard form a circle.

            if is_distance_OK == False:
                continue

            is_dangerous_OK = obtain_enemy_around_position(board, ship.position,
                                                           size=2)  # <--- do not want to build with enemy aside

            if is_distance_OK and is_dangerous_OK:
                num_my_ship, mean_halite, num_cargo, my_halite_density = obtain_my_ships_in_position(board, ship.position, size=2)
                if mean_halite > 5 and my_halite_density >= 50:  # <-- do not want to build with no halite surround
                    # try to make your shipyard towards the center
                    curr_score = (
                                num_my_ship * 20 + 50 * mean_halite + num_cargo * 0.5 + 50 * (
                                abs(10 - ship.position[0]) + abs(10 - ship.position[1])) + 300 * num_dist_le_7
                    )
                    ship_convert_score.append((ship, curr_score))

        if len(ship_convert_score):
            print(board.step, [w[0].position for w in ship_convert_score])
            ship_convert_score = sorted(ship_convert_score, key=lambda x: x[1], reverse=True)
            curr_selected_ship = ship_convert_score[0][0]
            if curr_selected_ship.halite + turn.total_halite > 500:
                print('build shipyard on purpose:')
                converts(curr_selected_ship, turn)
                last_convert_timer = board.step


def is_local_min(position):
    row, col = translate_pos(position)
    local_cargos = slice(cargo_matrix, row - 1, row + 1, col - 1, col + 1)
    local_flags = slice(turn.flag_matrix, row - 1, row + 1, col - 1, col + 1) == 2
    if np.sum(local_flags) == 0:
        return True
    return np.nanmin(np.where(local_flags, local_cargos, np.nan)) == cargo_matrix[row, col]


def get_ship_halite(pos):
    ship = {ship.position: ship for ship in me.ships}.get(pos)
    if ship:
        return ship.halite
    else:
        return 10000

#################################################################
# ship movements
#################################################################

def ship_moves(board):
    global attackers
    global PROTECT_MODE
    # adj4_regular_check_on_high_halite_low_ship_density_region()

    ships = [ship for ship in me.ships if ship.next_action is None]
    # update ship_target
    assign_targets(board, ships)
    # For all ships without a target, we give them a random movement (we will check below if this
    actions = {}  # record actions for each ship
    for ship in ships:
        if ship.id in ship_target:
            a, delta = dirs_to(ship.position, ship_target[ship.id], size=size)
            actions[ship.id] = a
        else:
            actions[ship.id] = [random.choice(all_actions)]

    # attackers = {}
    for ship in sorted(ships, key=lambda ship: (1, ship.halite) if ship.id in set(
            guarders.values()) or ship.id in attackers or ship.id in missions else (0, ship.halite), reverse=True):
        action = None
        x = ship.position
        # generate matrix of places to attack and places to avoid
        avoid = make_avoidance_matrix(ship.halite)
        attack = make_attack_matrix(ship.halite)
        # see if there is a attack options
        action_list = actions[ship.id] + sorted([None] + all_actions, key=lambda action: nearest_shipyard(
            move(ship.position, action)))
        # see if we should add an attack diversion to our options
        # NOTE: we will avoid attacking a ship that is on the avoid spot - is this a good idea?
        for a in all_actions:
            m = move(x, a)
            row, col = translate_pos(m)
            if ((ship.id not in set(guarders.values()) and ship.id not in attackers)
                    and ((attack[m.y, m.x] and nearest_shipyard(m)[0] < 7)
                         or (ship.halite == 0 and nearest_shipyard(m)[0] < 7 and (
                                    turn.flag_matrix[row, col] == 4 or turn.flag_matrix[row, col] == 6))
                    )
            ):
                if ship.halite == 0 and nearest_shipyard(m)[0] < 7 and (
                        turn.flag_matrix[row, col] == 4 or turn.flag_matrix[row, col] == 6):
                    if ship.id not in attackers and m not in set(attackers.values()):
                        attackers[ship.id] = m
                print('ship id {} attacking {}'.format(ship.id, a))
                action_list.insert(0, a)
                break
        # now try the options, but don't bother repeating any
        action_list = remove_dups(action_list)
        found = False
        for a in action_list:
            m = move(x, a)
            if a is None and PROTECT_MODE and board.cells[m].halite > 0:
                continue
            if (avoid[m.y, m.x] and not guarders.get(m) == ship.id) and not (ship.id in attackers):
                print('ship id {} avoiding {}'.format(ship.id, a))
            if m not in turn.taken and (
            (not (avoid[m.y, m.x] and not guarders.get(m) == ship.id and not ship.id in attackers))):
                action = a
                found = True
                break
        if found:
            ship.next_action = action
            if board.step < MERGE_TIME:
                turn.taken[m] = 1

        else:
            best_action = None
            min_enemy_density = 10000
            min_halite_density = 10000
            min_dist2home = 10000
            for a in action_list:
                if a is None:
                    continue
                m = move(x, a)
                row, col = translate_pos(m)
                if m not in turn.taken and (is_local_min(m) or turn.flag_matrix[row, col] == 0):
                    action = a
                    enemy_flag = slice(turn.flag_matrix, row - 2, row + 2, col - 2, col + 2) == 2

                    invalid_points = np.argwhere(enemy_flag)
                    invalid_points = invalid_points[
                        np.abs((invalid_points - np.array(enemy_flag.shape) // 2)).sum(axis=1) > 2]
                    enemy_flag[invalid_points[:, 0], invalid_points[:, 1]] = False

                    enemy_cargo = slice(cargo_matrix, row - 2, row + 2, col - 2, col + 2)
                    enemy_density = np.logical_and((enemy_cargo <= ship.halite), enemy_flag)
                    enemy_density = np.sum(enemy_density)
                    curr_halite_density = halite_density.loc[row, col]
                    curr_dist2home = nearest_shipyard(m)
                    if (enemy_density, curr_halite_density, curr_dist2home) < (
                    min_enemy_density, min_halite_density, min_dist2home):
                        min_enemy_density = enemy_density
                        min_halite_density = curr_halite_density
                        min_dist2home = curr_dist2home
                        best_action = action
            ship.next_action = best_action
            if board.step < MERGE_TIME:
                turn.taken[m] = 1
        next_pos = move(x, ship.next_action)
        if board.cells[next_pos].shipyard:
            sy = board.cells[next_pos].shipyard
            if sy.next_action == ShipyardAction.SPAWN:
                sy.next_action = None
                turn.num_ships -= 1
                turn.total_halite += 500
                guarders[sy.position] = ship.id

enemy2sy_distances = []
enemy2sy_distances_trends = []
def trace_enemy_distances():
    """to my shipyards"""
    distances = {}
    distances_trend = {}
    for shipyard in me.shipyards:
        distances[shipyard.id] = {}
        distances_trend[shipyard.id] = {}
        for enemy_ship in board.ships.values():
            if enemy_ship.id == me.id:
                continue
            curr_dist = dist(enemy_ship.position, shipyard.position)
            distances[shipyard.id][enemy_ship.id] = curr_dist

            if len(enemy2sy_distances) > 1:
                prev_dists = enemy2sy_distances[-2].get(shipyard.id, {})
                if enemy_ship.id in prev_dists:
                    distances_trend[shipyard.id][enemy_ship.id] = (
                            curr_dist - prev_dists[enemy_ship.id])

    enemy2sy_distances.append(distances)
    enemy2sy_distances_trends.append(distances_trend)
    return

def get_approaching_enemy_ships():
    LAST_STEPS = 4
    if len(enemy2sy_distances_trends) < LAST_STEPS:
        return {}

    approaching_enemy_ships = {}
    for shipyard in me.shipyards:
        approaching_enemy_ships[shipyard.id] = []
        for ship in board.ships:
            if board.ships[ship].player_id == me.id:
                continue
            approaching = True
            for last_step in range(1, LAST_STEPS + 1):
                if enemy2sy_distances_trends[-last_step][shipyard.id].get(ship, 10000) >= 0:
                    approaching = False
                    break
            if approaching:
                approaching_enemy_ships[shipyard.id].append(ship)

    return approaching_enemy_ships

from sklearn.cluster import KMeans

def cluster_approaching_enemy_ships(approaching_enemy_ships):
    res = {}
    for shipyard in me.shipyards:
        enemy_ships = approaching_enemy_ships.get(shipyard.id, [])
        if len(enemy_ships) <= 3:
            continue
        positions = [list(board.ships[ship].position) for ship in enemy_ships]
        enemy_grouper = KMeans(3).fit(positions)
        group_labels = enemy_grouper.predict(positions)
        enemies_by_group = {label: [] for label in set(group_labels)}
        for ii in range(len(positions)):
            ship = enemy_ships[ii]
            label = group_labels[ii]
            enemies_by_group[label].append(ship)
        res[shipyard.id] = (enemies_by_group, enemy_grouper)
    return res

def detect_attacking_enemy_groups(approaching_enemy_ship_groups):
    global PROTECT_MODE
    attacking_enemy_groups = {}
    for shipyard in approaching_enemy_ship_groups:
        attacking_enemy_groups[shipyard] = []
        grouper = approaching_enemy_ship_groups[shipyard][1]
        for label, enemy_ships in approaching_enemy_ship_groups[shipyard][0].items():
            if len(enemy_ships) < 3:
                continue
            group_center = grouper.cluster_centers_[label]
            group_center = (round(group_center[0]), round(group_center[1]))
            if dist(group_center, board.shipyards[shipyard].position) > 5:
                continue
            attacking_enemy_groups[shipyard].append(([(ship, board.ships[ship].player_id) for ship in enemy_ships], group_center))
        if not attacking_enemy_groups[shipyard]:
            del attacking_enemy_groups[shipyard]
    if attacking_enemy_groups:
        print_('+' * 30, board.step, attacking_enemy_groups)
        PROTECT_MODE = True
    else:
        PROTECT_MODE = False
    return attacking_enemy_groups

def check_crushed_ship_players():
    global prev_board
    global board

    if not prev_board:
        return set()

    lost_empty_ships = {ii: set() for ii in range(4)}
    lost_map = np.ones((size, size)) * np.nan
    for ship in prev_board.ships.values():
        if ship.halite != 0:
            continue
        if ship.id not in board.ships and not board.cells[ship.position].shipyard:
            lost_empty_ships[ship.player_id].add(ship.id)
            lost_map[ship.position] = ship.player_id

    if not lost_empty_ships[me.id]:
        return set()

    risky_players = set()
    for ship in lost_empty_ships[me.id]:
        ship = prev_board.ships[ship]
        nearby_enemy_ships = get_enemies_around_point(prev_board, ship.position, 2, is_enemy=True)[1]
        nearby_ships = [enemy_ship.player_id for enemy_ship in nearby_enemy_ships
                        if enemy_ship.id in lost_empty_ships[enemy_ship.player_id]]
        risky_players = risky_players | set(nearby_ships)

    return risky_players

#################################################################
# agent:
# Returns the commands we send to our ships and shipyards, must be last function in file
#################################################################

board = None
noavd_players = set()

def agent(obs, config):
    tic = time.time()

    global size
    global start
    global prev_board
    global me
    global did_init
    global halite
    global halite_density
    global board
    global missions
    global cargo_matrix
    global cargo_ratios
    global ship_ratios
    global prev_convert
    global attackers
    global enemies_around_point
    global avoid_matrixes

    enemies_around_point = {}
    attackers = {}
    avoid_matrixes = {}

    # Do initialization 1 time
    start_step = time.time()
    if start is None:
        start = time.time()
    if not did_init:
        init(obs, config)
        did_init = True

    halite = np.array(obs['halite']).reshape(size, size)
    halite_density = get_dist_weighted_density(
        pd.DataFrame(halite), kernel_size=7)
    prev_board = board
    board = Board(obs, config)

    for player_id in check_crushed_ship_players():
        noavd_players.add(player_id)

    cargo_matrix = get_cargo_matrix()
    cargo_ratios = {}
    ship_ratios = {}

    me = board.current_player
    set_turn_data(board)
    print('==== step {} sim {}'.format(board.step, board.step + 1))
    print('ships {} shipyards {}'.format(turn.num_ships, turn.num_shipyards))
    print_enemy_ships(board)

    trace_enemy_distances()
    approaching_enemy_ships = get_approaching_enemy_ships()
    approaching_enemy_clusters = cluster_approaching_enemy_ships(approaching_enemy_ships)
    detect_attacking_enemy_groups(approaching_enemy_clusters)

    ship_converts(board)
    shipyard_actions()
    ship_moves(board)
    print_actions(board)
    print('time this turn: {:8.3f} total elapsed {:8.3f}'.format(time.time() - start_step, time.time() - start))
    if 'CONVERT' in set(me.next_actions.values()):
        prev_convert = board.step

    if board.step < MERGE_TIME:
        # adj: v3_21_adj22_feedback3_adjtrunc, comment out this to speed up
        # adj: v3_21_adj22_feedback3_full_adj6, bring this back
        my_ships = {ship.id: ship for ship in me.ships}
        sy_pos = set([sy.position for sy in me.shipyards])
        pos2ship = {ship.position: ship for ship in me.ships}
        ship_pos = set([ship.position for ship in me.ships])
        guarded = sy_pos.intersection(ship_pos)
        if guarded:
            for ship_id, action in me.next_actions.items():
                if ship_id not in my_ships:
                    continue
                old_pos = my_ships[ship_id].position
                new_pos = move(old_pos, action)
                if new_pos in guarded:
                    guarder_ship = pos2ship[new_pos].id
                    if guarder_ship not in me.next_actions:
                        me.next_actions[guarder_ship] = dirs_to(new_pos, old_pos, size=21)[0]

        actions_str2obj = {'SOUTH': ShipAction.SOUTH,
                           'WEST': ShipAction.WEST,
                           'NORTH': ShipAction.NORTH,
                           'EAST': ShipAction.EAST}

        for ii in range(2):
            changed = False
            conflicts = defaultdict(list)
            # for ship, action in me.next_actions.items():
            for ship_id, ship in my_ships.items():
                if ship_id in me.next_actions:
                    action = me.next_actions[ship_id]
                    if action == 'CONVERT':
                        continue
                    new_pos = move(ship.position, actions_str2obj[action])
                else:
                    new_pos = ship.position
                conflicts[new_pos].append(ship)

            for newpos, ships in conflicts.items():
                if len(ships) > 1:
                    changed = True
                    new_conflicts = defaultdict(list)
                    ships.sort(key=lambda ship: ship.halite)
                    for ii, ship in enumerate(ships[:-1]):
                        for action in all_actions:
                            nnpos = move(ship.position, action)
                            if nnpos not in conflicts and nnpos not in new_conflicts:
                                ship.next_action = action
                                new_conflicts[nnpos].append(ship)
                                break

            if not changed:
                break

    # final cordinate
    for ship in me.ships:
        m = move(ship.position, ship.next_action)
        sy = board.cells[m].shipyard
        if sy and sy.next_action == ShipyardAction.SPAWN:
            sy.next_action = None

    # # =====================================================================================
    # my_ships = {ship.id: ship for ship in me.ships}
    # sy_pos = set([sy.position for sy in me.shipyards])
    # pos2ship = {ship.position: ship for ship in me.ships}
    # ship_pos = set([ship.position for ship in me.ships])
    # guarded = sy_pos.intersection(ship_pos)
    #
    # new_positions = {ship: move(board.ships[ship].position, action) for ship, action in me.next_actions.items() if  ship in my_ships}
    # len(new_positions.values())
    # len(set(new_positions.values()))
    #
    # def view(ships):
    #
    #   print_('ship positions: ', {ship.id: translate_pos(ship.position) for ship in me.ships})
    #   print_('shipyard positions: ', {sy.id: translate_pos(sy.position) for sy in me.shipyards})
    #
    #   ships_map = [[''] * size for _ in range(size)]
    #   for ship in ships:
    #     row, col = translate_pos(ship.position)
    #     if ship.position in sy_pos:
    #       ships_map[row][col] = f"**{ship.id}: {me.next_actions.get(ship.id)}"
    #     else:
    #       ships_map[row][col] = f"##({ship.id},{ship.halite}): {me.next_actions.get(ship.id)}"
    #   for shipyard in sy_pos:
    #     row, col = translate_pos(shipyard)
    #     ships_map[row][col] = '*****' + ' & ' + ships_map[row][col]
    #
    #   for ship in board.ships.values():
    #     if ship.player_id != me.id:
    #       row, col = translate_pos(ship.position)
    #       ships_map[row][col] = f'==es({ship.id},{ship.halite})'
    #
    #   for sy in board.shipyards.values():
    #     if sy.player_id != me.id:
    #       row, col = translate_pos(sy.position)
    #       ships_map[row][col] = '==EY'
    #
    #   ships_map_orig = pd.DataFrame(ships_map)
    #
    #   col_widths = defaultdict(int)
    #   for jj in range(size):
    #     for ii in range(size):
    #       col_widths[jj] = max(col_widths[jj], len(ships_map[ii][jj]))
    #
    #   for jj in range(size):
    #     if col_widths[jj] == 0:
    #       continue
    #     for ii in range(size):
    #       ships_map[ii][jj] = '|' + ships_map[ii][jj] + ' ' * (col_widths[jj] - len(ships_map[ii][jj])) + '|'
    #   ships_map = pd.DataFrame(ships_map)
    #
    #   ships_map.index = inds
    #   ships_map_orig.index = inds
    #   return ships_map_orig, ships_map
    #
    # print_(f'\n\nstep: {board.step}', '<' * 50)
    # ships_map_orig, ships_map = view(me.ships)
    # print_(ships_map)
    # print_(f'step: {board.step}', '>' * 50 + '\n')
    # # =====================================================================================
    # print_('advnoadv', '=' * 30, board.step, time.time() - tic)
    # print_(me.next_actions)
    return me.next_actions
