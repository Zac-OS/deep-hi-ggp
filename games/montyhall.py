# dumping propnet for /Users/zac/work/comp/thesis/code/rulesheets_gdl2/montyhall.gdl
# number of roles 2

from constants import *

roles = [
    'candidate',
    'random',
]

entries = (
    (0, -1, OR, [70, 33, 91], [30]),
    (1, -1, OR, [7, 8], [26]),
    (2, -1, NOT, [100], [11, 91, 85]),
    (3, -1, TRANSITION, [104], [120]),
    (4, -1, OR, [14, 49], [36]),
    (5, -1, PROPOSITION, [12], [], goal, '( goal random 100 )'),
    (6, -1, PROPOSITION, [49], [], legal, '( legal random noop )'),
    (7, -1, PROPOSITION, [41], [1], other, 'anon'),
    (8, -1, PROPOSITION, [], [53, 1, 39, 29], init, 'init'),
    (9, -1, PROPOSITION, [], [20], input, '( does random ( open_door 3 ) )'),
    (10, -1, TRANSITION, [68], [57]),
    (11, -1, AND, [2, 57], [37]),
    (12, 1, CONSTANT, [], [5]),
    (13, -1, PROPOSITION, [], [102], input, '( does candidate ( choose 3 ) )'),
    (14, -1, PROPOSITION, [44], [4, 38, 88, 111, 94], base, '( true ( step 2 ) )'),
    (15, -1, PROPOSITION, [], [109], input, '( does random ( hide_car 3 ) )'),
    (16, -1, PROPOSITION, [30], [66], other, 'anon'),
    (17, -1, TRANSITION, [99], [98]),
    (18, -1, NOT, [82], [41]),
    (19, -1, PROPOSITION, [121], [], legal, '( legal random ( hide_car 2 ) )'),
    (20, -1, NOT, [9], [83]),
    (21, -1, PROPOSITION, [117], [], sees, '( sees candidate ( car 1 ) )'),
    (22, -1, PROPOSITION, [77], [], sees, '( sees candidate ( car 2 ) )'),
    (23, -1, PROPOSITION, [], [25], input, '( does random ( hide_car 1 ) )'),
    (24, -1, TRANSITION, [53], [105]),
    (25, -1, OR, [98, 23], [99]),
    (26, -1, TRANSITION, [1], [101]),
    (27, -1, NOT, [116], [107, 111]),
    (28, -1, TRANSITION, [40], [116]),
    (29, -1, TRANSITION, [8], [61]),
    (30, -1, PROPOSITION, [0], [117, 16], other, '( next_chosen 1 )'),
    (31, -1, PROPOSITION, [37], [68, 77], other, '( next_chosen 2 )'),
    (32, -1, OR, [60, 86], [55]),
    (33, -1, AND, [100, 92, 64], [0]),
    (34, -1, TRANSITION, [106], [69]),
    (35, -1, AND, [61, 64], [108]),
    (36, -1, PROPOSITION, [4], [], legal, '( legal candidate noop )'),
    (37, -1, OR, [11, 59, 122], [31]),
    (38, -1, AND, [14, 92, 75, 64], [81]),
    (39, -1, OR, [8, 95], [56]),
    (40, -1, PROPOSITION, [109], [28], other, 'anon'),
    (41, -1, AND, [101, 18], [7]),
    (42, -1, AND, [57, 86], [103]),
    (43, -1, AND, [61, 105], [80]),
    (44, -1, TRANSITION, [72], [14]),
    (45, -1, AND, [52, 57], [67]),
    (46, -1, PROPOSITION, [67], [], goal, '( goal candidate 0 )'),
    (47, -1, AND, [75, 79], [67]),
    (48, -1, PROPOSITION, [111], [], legal, '( legal random ( open_door 3 ) )'),
    (49, -1, PROPOSITION, [58], [4, 6, 114, 104, 77, 117, 78], base, '( true ( step 3 ) )'),
    (50, -1, PROPOSITION, [118], [], legal, '( legal candidate ( choose 1 ) )'),
    (51, -1, AND, [74, 64], [95]),
    (52, -1, NOT, [86], [45, 88]),
    (53, -1, OR, [54, 8], [24]),
    (54, -1, PROPOSITION, [83], [53], other, 'anon'),
    (55, -1, PROPOSITION, [32], [110], other, 'anon'),
    (56, -1, TRANSITION, [39], [64]),
    (57, -1, PROPOSITION, [10], [11, 45, 42, 123], base, '( true ( chosen 2 ) )'),
    (58, -1, TRANSITION, [94], [49]),
    (59, -1, AND, [100, 101, 123], [37]),
    (60, -1, PROPOSITION, [], [32], input, '( does random ( hide_car 2 ) )'),
    (61, -1, PROPOSITION, [29], [71, 72, 89, 121, 118, 43, 35], base, '( true ( step 1 ) )'),
    (62, -1, PROPOSITION, [], [], input, '( does random switch )'),
    (63, -1, AND, [69, 116], [103]),
    (64, -1, PROPOSITION, [56], [38, 33, 51, 118, 35], base, '( true ( closed 1 ) )'),
    (65, -1, PROPOSITION, [88], [], legal, '( legal random ( open_door 2 ) )'),
    (66, -1, TRANSITION, [16], [79]),
    (67, -1, OR, [107, 45, 47], [46]),
    (68, -1, PROPOSITION, [31], [10], other, 'anon'),
    (69, -1, PROPOSITION, [34], [107, 90, 63, 85], base, '( true ( chosen 3 ) )'),
    (70, -1, PROPOSITION, [], [0], input, '( does candidate ( choose 1 ) )'),
    (71, -1, AND, [61, 105], [96]),
    (72, -1, PROPOSITION, [61], [44], other, 'anon'),
    (73, -1, PROPOSITION, [], [], input, '( does random noop )'),
    (74, -1, NOT, [113], [51]),
    (75, -1, NOT, [98], [38, 47]),
    (76, -1, AND, [98, 79], [103]),
    (77, -1, AND, [31, 49, 86], [22]),
    (78, -1, AND, [49, 116, 93], [97]),
    (79, -1, PROPOSITION, [66], [47, 91, 92, 76], base, '( true ( chosen 1 ) )'),
    (80, -1, PROPOSITION, [43], [], legal, '( legal random ( hide_car 3 ) )'),
    (81, -1, PROPOSITION, [38], [], legal, '( legal random ( open_door 1 ) )'),
    (82, -1, PROPOSITION, [], [18], input, '( does random ( open_door 2 ) )'),
    (83, -1, AND, [105, 20], [54]),
    (84, -1, PROPOSITION, [103], [], goal, '( goal candidate 100 )'),
    (85, -1, AND, [2, 69], [102]),
    (86, -1, PROPOSITION, [110], [52, 32, 77, 42], base, '( true ( car 2 ) )'),
    (87, -1, AND, [100, 90, 105], [102]),
    (88, -1, AND, [52, 101, 14, 123], [65]),
    (89, -1, AND, [101, 61], [115]),
    (90, -1, NOT, [69], [87, 111]),
    (91, -1, AND, [2, 79], [0]),
    (92, -1, NOT, [79], [38, 33]),
    (93, -1, PROPOSITION, [102], [78, 106], other, '( next_chosen 3 )'),
    (94, -1, PROPOSITION, [14], [58], other, 'anon'),
    (95, -1, PROPOSITION, [51], [39], other, 'anon'),
    (96, -1, PROPOSITION, [71], [], legal, '( legal candidate ( choose 3 ) )'),
    (97, -1, PROPOSITION, [78], [], sees, '( sees candidate ( car 3 ) )'),
    (98, -1, PROPOSITION, [17], [25, 75, 76, 117], base, '( true ( car 1 ) )'),
    (99, -1, PROPOSITION, [25], [17], other, 'anon'),
    (100, -1, PROPOSITION, [], [2, 87, 59, 33], input, '( does candidate switch )'),
    (101, -1, PROPOSITION, [26], [59, 89, 88, 121, 41], base, '( true ( closed 2 ) )'),
    (102, -1, OR, [87, 13, 85], [93]),
    (103, -1, OR, [76, 63, 42], [84]),
    (104, -1, PROPOSITION, [49], [3], other, 'anon'),
    (105, -1, PROPOSITION, [24], [87, 71, 83, 111, 43], base, '( true ( closed 3 ) )'),
    (106, -1, PROPOSITION, [93], [34], other, 'anon'),
    (107, -1, AND, [69, 27], [67]),
    (108, -1, PROPOSITION, [35], [], legal, '( legal random ( hide_car 1 ) )'),
    (109, -1, OR, [15, 116], [40]),
    (110, -1, TRANSITION, [55], [86]),
    (111, -1, AND, [14, 90, 27, 105], [48]),
    (112, -1, PROPOSITION, [120], [], terminal, 'terminal'),
    (113, -1, PROPOSITION, [], [74], input, '( does random ( open_door 1 ) )'),
    (114, -1, PROPOSITION, [49], [], legal, '( legal candidate switch )'),
    (115, -1, PROPOSITION, [89], [], legal, '( legal candidate ( choose 2 ) )'),
    (116, -1, PROPOSITION, [28], [109, 27, 63, 78], base, '( true ( car 3 ) )'),
    (117, -1, AND, [30, 98, 49], [21]),
    (118, -1, AND, [61, 64], [50]),
    (119, -1, PROPOSITION, [], [], input, '( does candidate noop )'),
    (120, -1, PROPOSITION, [3], [112], base, '( true ( step 4 ) )'),
    (121, -1, AND, [101, 61], [19]),
    (122, -1, PROPOSITION, [], [37], input, '( does candidate ( choose 2 ) )'),
    (123, -1, NOT, [57], [59, 88]),
)

# DONE
