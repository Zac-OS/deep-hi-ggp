# dumping propnet for /Users/zac/work/comp/thesis/code/rulesheets_gdl2/blindtictactoeXbias.gdl
# number of roles 3

from constants import *

roles = [
    'x',
    'o',
    'random',
]

entries = (
    (0, -1, PROPOSITION, [200], [], goal, '( goal x 0 )'),
    (1, -1, AND, [414, 534], [591]),
    (2, -1, OR, [87, 506], [432]),
    (3, -1, AND, [66, 304, 199], [575]),
    (4, -1, AND, [606, 87, 557, 506], [276]),
    (5, -1, OR, [550, 516], [498]),
    (6, -1, AND, [66, 587, 304], [575]),
    (7, -1, AND, [187, 373], [614]),
    (8, -1, PROPOSITION, [15], [88, 595, 201, 193, 338, 76], other, '( next_tmp3 3 )'),
    (9, -1, AND, [389, 460], [482]),
    (10, -1, NOT, [337], [429]),
    (11, -1, OR, [579, 52], [181]),
    (12, -1, AND, [28, 380, 578], [86]),
    (13, -1, PROPOSITION, [404], [], other, '( next ( cell 2 3 xSecondChance ) )'),
    (14, -1, PROPOSITION, [190], [313], other, '( next ( cell 3 2 b ) )'),
    (15, -1, OR, [171, 433, 579], [8]),
    (16, -1, PROPOSITION, [555], [], legal, '( legal x ( mark 1 2 ) )'),
    (17, -1, OR, [66, 606, 494, 234, 151, 187, 446, 136, 147], [574]),
    (18, -1, AND, [494, 134], [218]),
    (19, -1, OR, [215, 600], [164]),
    (20, -1, PROPOSITION, [381], [305], other, '( next ( cell 2 1 x ) )'),
    (21, -1, AND, [171, 234], [611]),
    (22, -1, PROPOSITION, [614], [301, 110], other, '( sees_tmp1 3 o )'),
    (23, -1, AND, [606, 87], [302]),
    (24, -1, PROPOSITION, [453], [522, 319, 284, 470], base, '( true ( cell 1 1 o ) )'),
    (25, -1, AND, [494, 134, 64], [381]),
    (26, -1, NOT, [296], [461]),
    (27, -1, AND, [87, 606, 321], [99]),
    (28, -1, PROPOSITION, [186], [604, 12, 284, 100], base, '( true ( cell 1 3 o ) )'),
    (29, -1, TRANSITION, [133], [613]),
    (30, -1, TRANSITION, [420], [97]),
    (31, -1, OR, [168, 185, 457, 554, 566, 440, 338], [247]),
    (32, -1, PROPOSITION, [99], [530], other, '( next ( cell 2 2 o ) )'),
    (33, -1, AND, [66, 433], [332]),
    (34, -1, AND, [183, 196], [591]),
    (35, -1, PROPOSITION, [270], [], legal, '( legal x ( mark 1 1 ) )'),
    (36, -1, PROPOSITION, [82], [336], other, '( next ( cell 3 3 o ) )'),
    (37, -1, NOT, [315], [503]),
    (38, -1, AND, [151, 445], [512]),
    (39, -1, PROPOSITION, [591], [], sees, '( sees o ok )'),
    (40, -1, NOT, [181], [250]),
    (41, -1, PROPOSITION, [408], [419], other, '( next ( cell 3 2 o ) )'),
    (42, -1, PROPOSITION, [58], [254, 174], base, '( true ( tried x 3 2 ) )'),
    (43, -1, PROPOSITION, [], [113, 413, 345, 91, 415, 313, 527, 316, 103], init, 'init'),
    (44, -1, AND, [569, 442], [482]),
    (45, -1, NOT, [94], [205]),
    (46, -1, AND, [557, 303], [482]),
    (47, -1, OR, [166, 215, 472], [540]),
    (48, -1, AND, [494, 472, 532], [263]),
    (49, -1, AND, [66, 433], [128]),
    (50, -1, TRANSITION, [294], [118]),
    (51, -1, AND, [596, 214, 147], [598]),
    (52, -1, PROPOSITION, [], [545, 582, 331, 495, 497, 407, 548, 476, 227, 111, 253, 11, 56, 233, 365, 367, 314, 529], input, '( does x ( mark 3 3 ) )'),
    (53, -1, TRANSITION, [164], [600]),
    (54, -1, OR, [255, 579], [280]),
    (55, -1, AND, [596, 532, 147], [598]),
    (56, -1, AND, [446, 52], [236]),
    (57, -1, PROPOSITION, [519], [510], other, '( next ( tried o 1 3 ) )'),
    (58, -1, TRANSITION, [292], [42]),
    (59, -1, AND, [596, 147], [216]),
    (60, -1, PROPOSITION, [444], [198], other, '( next ( tried o 3 2 ) )'),
    (61, -1, AND, [187, 535, 373], [408]),
    (62, -1, OR, [115, 445, 304], [535]),
    (63, -1, PROPOSITION, [580], [525], other, '( row 3 o )'),
    (64, -1, PROPOSITION, [421], [340, 564, 323, 476, 25, 206], other, '( next_tmp1 1 )'),
    (65, -1, AND, [377, 327, 185], [232]),
    (66, -1, PROPOSITION, [148], [514, 378, 105, 3, 563, 6, 165, 500, 360, 384, 49, 463, 33, 387, 272, 521, 17, 483, 124, 556, 126], base, '( true ( cell 1 3 b ) )'),
    (67, -1, AND, [223, 462, 584], [117]),
    (68, -1, TRANSITION, [597], [107]),
    (69, -1, PROPOSITION, [419], [580, 71, 408], base, '( true ( cell 3 2 o ) )'),
    (70, -1, PROPOSITION, [598], [508], other, '( next ( cell 1 2 o ) )'),
    (71, -1, AND, [69, 380, 293], [222]),
    (72, -1, AND, [494, 134], [217]),
    (73, -1, PROPOSITION, [480], [], legal, '( legal o ( mark 2 3 ) )'),
    (74, -1, PROPOSITION, [361], [], terminal, 'terminal'),
    (75, -1, AND, [136, 228], [358]),
    (76, -1, AND, [281, 136, 8], [326]),
    (77, -1, TRANSITION, [204], [578]),
    (78, -1, OR, [107, 134], [597]),
    (79, -1, AND, [281, 136], [218]),
    (80, 1, CONSTANT, [], [354, 344, 507, 101]),
    (81, -1, AND, [171, 234, 321], [371]),
    (82, -1, OR, [375, 418, 163, 264, 602, 253], [36]),
    (83, -1, PROPOSITION, [450], [113], other, '( next ( cell 1 1 b ) )'),
    (84, -1, PROPOSITION, [341], [], goal, '( goal x 100 )'),
    (85, -1, AND, [166, 151, 196, 445], [522]),
    (86, -1, OR, [12, 319], [177]),
    (87, -1, PROPOSITION, [], [179, 27, 454, 351, 149, 2, 4, 308, 95, 501, 388, 411, 143, 399, 23, 543, 243], input, '( does o ( mark 2 2 ) )'),
    (88, -1, AND, [114, 187, 8], [523]),
    (89, -1, NOT, [106], [450]),
    (90, -1, AND, [115, 557, 596, 147], [353]),
    (91, -1, OR, [43, 546], [318]),
    (92, -1, AND, [460, 197], [482]),
    (93, -1, AND, [150, 414], [591]),
    (94, -1, PROPOSITION, [265], [45], other, '( marked 1 2 )'),
    (95, -1, OR, [87, 115, 114, 596, 506, 373], [561]),
    (96, -1, AND, [225, 295], [482]),
    (97, -1, PROPOSITION, [30], [351, 423], base, '( true ( tried o 2 2 ) )'),
    (98, -1, AND, [115, 596, 435, 147], [353]),
    (99, -1, OR, [27, 388, 380, 543, 501, 243], [32]),
    (100, -1, OR, [272, 28, 378, 563, 126, 463], [153]),
    (101, -1, PROPOSITION, [80], [], legal, '( legal random ( tiebreak o ) )'),
    (102, -1, TRANSITION, [498], [550]),
    (103, -1, OR, [43, 358], [428]),
    (104, -1, NOT, [432], [526]),
    (105, -1, AND, [66, 304, 585], [575]),
    (106, -1, PROPOSITION, [109], [89], other, '( marked 1 1 )'),
    (107, -1, PROPOSITION, [68], [78, 210], base, '( true ( tried x 2 1 ) )'),
    (108, -1, PROPOSITION, [410], [242, 270], base, '( true ( tried x 1 1 ) )'),
    (109, -1, OR, [166, 445], [106]),
    (110, -1, AND, [22, 197], [591]),
    (111, -1, OR, [330, 52], [339]),
    (112, -1, PROPOSITION, [594], [604, 202, 371], base, '( true ( cell 2 3 o ) )'),
    (113, -1, OR, [43, 83], [465]),
    (114, -1, PROPOSITION, [], [220, 129, 88, 456, 288, 95, 502, 254, 138, 340, 154, 233, 158, 368, 314, 278, 240, 317], input, '( does x ( mark 3 2 ) )'),
    (115, -1, PROPOSITION, [], [220, 201, 251, 90, 265, 285, 290, 95, 98, 449, 325, 62, 274, 119, 417, 400, 144, 369], input, '( does x ( mark 1 2 ) )'),
    (116, -1, AND, [573, 382], [577]),
    (117, -1, OR, [167, 67], [307]),
    (118, -1, PROPOSITION, [50], [480, 207], base, '( true ( tried o 2 3 ) )'),
    (119, -1, OR, [166, 115, 596, 433, 445, 304], [197]),
    (120, -1, AND, [516, 234], [488]),
    (121, -1, OR, [517, 400, 502], [569]),
    (122, -1, PROPOSITION, [523], [551], other, '( next ( cell 3 2 x ) )'),
    (123, -1, OR, [281, 134, 445], [321]),
    (124, -1, AND, [66, 553], [277]),
    (125, -1, PROPOSITION, [437], [524], other, '( next ( tried o 2 1 ) )'),
    (126, -1, AND, [66, 433, 532], [100]),
    (127, -1, AND, [215, 136], [128]),
    (128, -1, OR, [262, 504, 350, 335, 143, 542, 499, 127, 49], [183]),
    (129, -1, AND, [114, 435, 187, 373], [523]),
    (130, -1, AND, [494, 214, 472], [263]),
    (131, -1, PROPOSITION, [423], [], legal, '( legal o ( mark 2 2 ) )'),
    (132, -1, AND, [494, 472], [230]),
    (133, -1, PROPOSITION, [263], [29], other, '( next ( cell 2 1 o ) )'),
    (134, -1, PROPOSITION, [], [349, 454, 441, 248, 405, 72, 266, 268, 137, 477, 78, 18, 571, 193, 123, 469, 25, 244], input, '( does x ( mark 2 1 ) )'),
    (135, -1, AND, [171, 234], [302]),
    (136, -1, PROPOSITION, [428], [184, 334, 562, 564, 75, 518, 310, 76, 605, 607, 79, 297, 298, 505, 17, 362, 393, 576, 127, 245, 246], base, '( true ( cell 3 1 b ) )'),
    (137, -1, OR, [506, 516, 134], [273]),
    (138, -1, AND, [114, 587, 187], [523]),
    (139, -1, PROPOSITION, [328], [345], other, '( next ( cell 2 1 b ) )'),
    (140, -1, AND, [520, 389], [591]),
    (141, -1, PROPOSITION, [526], [415], other, '( next ( cell 2 2 b ) )'),
    (142, -1, AND, [171, 234, 516, 435], [430]),
    (143, -1, AND, [606, 87], [128]),
    (144, -1, AND, [115, 557, 596, 147], [533]),
    (145, -1, NOT, [589], [190]),
    (146, -1, TRANSITION, [339], [330]),
    (147, -1, PROPOSITION, [487], [201, 251, 90, 205, 285, 290, 51, 98, 504, 449, 55, 17, 59, 274, 366, 175, 417, 400, 144, 439, 369], base, '( true ( cell 1 2 b ) )'),
    (148, -1, TRANSITION, [527], [66]),
    (149, -1, AND, [606, 87, 557, 506], [267]),
    (150, -1, PROPOSITION, [216], [93, 511], other, '( sees_tmp3 2 o )'),
    (151, -1, PROPOSITION, [465], [335, 457, 287, 566, 338, 603, 168, 466, 467, 256, 169, 17, 155, 450, 436, 38, 397, 194, 554, 85, 440], base, '( true ( cell 1 1 b ) )'),
    (152, -1, TRANSITION, [568], [475]),
    (153, -1, PROPOSITION, [100], [186], other, '( next ( cell 1 3 o ) )'),
    (154, -1, AND, [114, 540, 187], [523]),
    (155, -1, AND, [166, 557, 151, 445], [238]),
    (156, -1, PROPOSITION, [430], [422], other, '( next ( cell 2 3 x ) )'),
    (157, -1, PROPOSITION, [342], [], legal, '( legal o ( mark 3 2 ) )'),
    (158, -1, AND, [114, 187, 196, 373], [408]),
    (159, -1, AND, [494, 472], [302]),
    (160, -1, OR, [171, 516], [547]),
    (161, -1, AND, [606, 506, 540], [267]),
    (162, -1, PROPOSITION, [576], [], other, '( next ( cell 3 1 xSecondChance ) )'),
    (163, -1, PROPOSITION, [336], [604, 580, 319, 82], base, '( true ( cell 3 3 o ) )'),
    (164, -1, PROPOSITION, [19], [53], other, '( next ( tried o 3 1 ) )'),
    (165, -1, AND, [66, 304], [512]),
    (166, -1, PROPOSITION, [], [421, 47, 335, 287, 109, 566, 477, 603, 169, 479, 155, 119, 436, 397, 194, 554, 85], input, '( does o ( mark 1 1 ) )'),
    (167, -1, AND, [573, 382], [117]),
    (168, -1, AND, [151, 445, 199], [31]),
    (169, -1, AND, [166, 273, 151], [522]),
    (170, -1, AND, [197, 295], [482]),
    (171, -1, PROPOSITION, [], [160, 454, 350, 497, 404, 583, 424, 135, 207, 271, 15, 396, 142, 21, 399, 81, 329], input, '( does o ( mark 2 3 ) )'),
    (172, -1, AND, [389, 406], [482]),
    (173, -1, PROPOSITION, [258], [], legal, '( legal o ( mark 1 2 ) )'),
    (174, -1, NOT, [42], [572]),
    (175, -1, AND, [596, 147], [332]),
    (176, -1, OR, [215, 579, 373], [585]),
    (177, -1, PROPOSITION, [86], [525], other, '( diagonal o )'),
    (178, -1, OR, [605, 607, 334, 578, 310, 246], [204]),
    (179, -1, AND, [606, 87], [216]),
    (180, -1, OR, [466, 290, 500], [406]),
    (181, -1, PROPOSITION, [11], [40], other, '( marked 3 3 )'),
    (182, -1, OR, [304, 565], [455]),
    (183, -1, PROPOSITION, [128], [34], other, '( sees_tmp5 o )'),
    (184, -1, AND, [281, 136], [488]),
    (185, -1, PROPOSITION, [431], [412, 65, 359, 31], base, '( true ( cell 1 1 x ) )'),
    (186, -1, TRANSITION, [153], [28]),
    (187, -1, PROPOSITION, [425], [129, 88, 456, 288, 190, 7, 499, 502, 409, 138, 208, 340, 154, 211, 17, 61, 158, 278, 240, 317, 261], base, '( true ( cell 3 2 b ) )'),
    (188, -1, AND, [386, 435], [482]),
    (189, -1, PROPOSITION, [510], [320, 519], base, '( true ( tried o 1 3 ) )'),
    (190, -1, AND, [187, 145], [14]),
    (191, -1, TRANSITION, [345], [494]),
    (192, -1, NOT, [550], [347]),
    (193, -1, AND, [494, 134, 8], [381]),
    (194, -1, AND, [166, 151, 532], [522]),
    (195, -1, NOT, [565], [471]),
    (196, -1, PROPOSITION, [], [449, 388, 34, 248, 378, 396, 158, 85, 310, 253], input, '( does random ( tiebreak o ) )'),
    (197, -1, PROPOSITION, [119], [586, 170, 92, 110], other, '( sees_tmp9 1 )'),
    (198, -1, TRANSITION, [60], [219]),
    (199, -1, PROPOSITION, [411], [168, 571, 363, 3, 367, 245], other, '( next_tmp3 2 )'),
    (200, -1, AND, [223, 382], [0]),
    (201, -1, AND, [115, 147, 8], [353]),
    (202, -1, AND, [112, 380, 613], [592]),
    (203, -1, AND, [569, 414], [482]),
    (204, -1, PROPOSITION, [178], [77], other, '( next ( cell 3 1 o ) )'),
    (205, -1, AND, [45, 147], [599]),
    (206, -1, AND, [606, 506, 64], [267]),
    (207, -1, OR, [171, 118], [294]),
    (208, -1, AND, [273, 187, 373], [408]),
    (209, -1, PROPOSITION, [305], [412, 381, 374], base, '( true ( cell 2 1 x ) )'),
    (210, -1, NOT, [107], [468]),
    (211, -1, AND, [187, 373], [216]),
    (212, -1, TRANSITION, [455], [565]),
    (213, -1, AND, [496, 414], [482]),
    (214, -1, PROPOSITION, [227], [130, 51, 409, 501, 603, 246], other, '( next_tmp7 3 )'),
    (215, -1, PROPOSITION, [], [47, 334, 518, 477, 310, 605, 607, 298, 19, 362, 257, 393, 576, 176, 314, 127, 246], input, '( does o ( mark 3 1 ) )'),
    (216, -1, OR, [179, 211, 59], [150]),
    (217, -1, OR, [481, 451, 72], [460]),
    (218, -1, OR, [79, 256, 18], [438]),
    (219, -1, PROPOSITION, [198], [342, 444], base, '( true ( tried o 3 2 ) )'),
    (220, -1, OR, [114, 115, 506], [416]),
    (221, -1, PROPOSITION, [371], [594], other, '( next ( cell 2 3 o ) )'),
    (222, -1, PROPOSITION, [71], [525], other, '( column 2 o )'),
    (223, -1, NOT, [573], [67, 235, 485, 200]),
    (224, -1, PROPOSITION, [490], [], legal, '( legal o ( mark 3 1 ) )'),
    (225, -1, PROPOSITION, [454], [301, 241, 96, 538], other, '( sees_tmp9 2 )'),
    (226, -1, TRANSITION, [610], [333]),
    (227, -1, OR, [516, 304, 52], [214]),
    (228, -1, NOT, [515], [75]),
    (229, -1, PROPOSITION, [302], [586, 612], other, '( sees_tmp1 2 o )'),
    (230, -1, OR, [132, 436, 518], [474]),
    (231, -1, PROPOSITION, [608], [283, 267, 259, 359, 374], base, '( true ( cell 2 2 x ) )'),
    (232, -1, PROPOSITION, [65], [492], other, '( row 1 x )'),
    (233, -1, OR, [114, 281, 52], [532]),
    (234, -1, PROPOSITION, [318], [350, 404, 443, 583, 424, 323, 135, 324, 448, 271, 299, 17, 363, 451, 120, 396, 142, 21, 486, 81, 329], base, '( true ( cell 2 3 b ) )'),
    (235, -1, AND, [223, 382], [300]),
    (236, -1, OR, [56, 562, 317], [295]),
    (237, -1, AND, [494, 472, 535], [263]),
    (238, -1, PROPOSITION, [155], [], other, '( next ( cell 1 1 xSecondChance ) )'),
    (239, -1, AND, [494, 416, 472], [263]),
    (240, -1, AND, [114, 187], [488]),
    (241, -1, AND, [406, 225], [482]),
    (242, -1, OR, [108, 445], [352]),
    (243, -1, AND, [87, 606, 532], [99]),
    (244, -1, AND, [494, 134], [488]),
    (245, -1, AND, [281, 136, 199], [326]),
    (246, -1, AND, [215, 214, 136], [178]),
    (247, -1, PROPOSITION, [31], [431], other, '( next ( cell 1 1 x ) )'),
    (248, -1, AND, [494, 472, 134, 196], [263]),
    (249, -1, AND, [446, 579], [611]),
    (250, -1, AND, [446, 40], [355]),
    (251, -1, AND, [115, 587, 147], [353]),
    (252, -1, PROPOSITION, [577], [], goal, '( goal o 50 )'),
    (253, -1, AND, [196, 446, 579, 52], [82]),
    (254, -1, OR, [114, 42], [292]),
    (255, -1, PROPOSITION, [391], [491, 54], base, '( true ( tried o 3 3 ) )'),
    (256, -1, AND, [151, 445], [218]),
    (257, -1, OR, [281, 215], [515]),
    (258, -1, NOT, [311], [173]),
    (259, -1, AND, [231, 377, 509], [464]),
    (260, -1, PROPOSITION, [522], [453], other, '( next ( cell 1 1 o ) )'),
    (261, -1, AND, [321, 187, 373], [408]),
    (262, -1, AND, [446, 579], [128]),
    (263, -1, OR, [248, 130, 613, 48, 237, 239], [133]),
    (264, -1, AND, [321, 446, 579], [82]),
    (265, -1, OR, [115, 596], [94]),
    (266, -1, AND, [494, 134, 585], [381]),
    (267, -1, OR, [161, 595, 231, 149, 308, 528, 206], [306]),
    (268, -1, AND, [494, 557, 472, 134], [403]),
    (269, -1, PROPOSITION, [552], [492], other, '( row 3 x )'),
    (270, -1, NOT, [108], [35]),
    (271, -1, AND, [171, 234, 532], [371]),
    (272, -1, AND, [66, 433, 321], [100]),
    (273, -1, PROPOSITION, [137], [208, 607, 169, 366, 563, 602], other, '( next_tmp5 2 )'),
    (274, -1, AND, [115, 147, 585], [353]),
    (275, -1, OR, [433, 304], [489]),
    (276, -1, PROPOSITION, [4], [], other, '( next ( cell 2 2 xSecondChance ) )'),
    (277, -1, PROPOSITION, [124], [527], other, '( next ( cell 1 3 b ) )'),
    (278, -1, AND, [114, 187], [512]),
    (279, -1, TRANSITION, [567], [311]),
    (280, -1, PROPOSITION, [54], [391], other, '( next ( tried o 3 3 ) )'),
    (281, -1, PROPOSITION, [], [184, 562, 564, 536, 477, 310, 76, 79, 297, 505, 362, 233, 257, 393, 576, 123, 314, 245], input, '( does x ( mark 3 1 ) )'),
    (282, -1, PROPOSITION, [459], [], goal, '( goal o 0 )'),
    (283, -1, AND, [231, 327, 475], [385]),
    (284, -1, AND, [28, 24, 293], [427]),
    (285, -1, AND, [115, 147], [488]),
    (286, -1, AND, [561, 474], [591]),
    (287, -1, AND, [166, 151], [332]),
    (288, -1, AND, [114, 557, 187, 373], [523]),
    (289, -1, TRANSITION, [390], [398]),
    (290, -1, AND, [115, 147], [180]),
    (291, -1, AND, [561, 534], [591]),
    (292, -1, PROPOSITION, [254], [58], other, '( next ( tried x 3 2 ) )'),
    (293, -1, PROPOSITION, [508], [71, 284, 598], base, '( true ( cell 1 2 o ) )'),
    (294, -1, PROPOSITION, [207], [50], other, '( next ( tried o 2 3 ) )'),
    (295, -1, PROPOSITION, [236], [170, 96], other, '( sees_tmp1 3 x )'),
    (296, -1, PROPOSITION, [544], [593, 26], base, '( true ( tried x 2 2 ) )'),
    (297, -1, AND, [281, 136], [512]),
    (298, -1, AND, [215, 136], [614]),
    (299, -1, AND, [234, 516], [512]),
    (300, -1, PROPOSITION, [235], [], goal, '( goal o 100 )'),
    (301, -1, AND, [22, 225], [591]),
    (302, -1, OR, [159, 135, 23], [229]),
    (303, -1, PROPOSITION, [488], [46], other, 'sees_tmp7'),
    (304, -1, PROPOSITION, [], [514, 182, 378, 497, 105, 3, 6, 227, 165, 500, 384, 387, 521, 62, 483, 275, 119, 556], input, '( does x ( mark 1 3 ) )'),
    (305, -1, TRANSITION, [20], [209]),
    (306, -1, PROPOSITION, [267], [608], other, '( next ( cell 2 2 x ) )'),
    (307, -1, PROPOSITION, [117], [], goal, '( goal x 50 )'),
    (308, -1, AND, [606, 87, 506, 435], [267]),
    (309, -1, NOT, [322], [328]),
    (310, -1, AND, [281, 215, 196, 136], [178]),
    (311, -1, PROPOSITION, [279], [560, 258], base, '( true ( tried o 1 2 ) )'),
    (312, -1, AND, [606, 506], [512]),
    (313, -1, OR, [43, 14], [425]),
    (314, -1, OR, [114, 215, 281, 579, 373, 52], [389]),
    (315, -1, PROPOSITION, [493], [479, 37], base, '( true ( tried o 1 1 ) )'),
    (316, -1, OR, [43, 355], [478]),
    (317, -1, AND, [114, 187], [236]),
    (318, -1, TRANSITION, [91], [234]),
    (319, -1, AND, [163, 380, 24], [86]),
    (320, -1, NOT, [189], [588]),
    (321, -1, PROPOSITION, [123], [27, 272, 264, 439, 81, 261], other, '( next_tmp7 1 )'),
    (322, -1, PROPOSITION, [349], [309], other, '( marked 2 1 )'),
    (323, -1, AND, [516, 234, 64], [430]),
    (324, -1, AND, [234, 447], [546]),
    (325, -1, OR, [115, 370], [376]),
    (326, -1, OR, [505, 362, 393, 564, 475, 76, 245], [568]),
    (327, -1, PROPOSITION, [434], [283, 575, 590, 65], base, '( true ( cell 1 3 x ) )'),
    (328, -1, AND, [494, 309], [139]),
    (329, -1, AND, [171, 234, 416], [371]),
    (330, -1, PROPOSITION, [146], [346, 111], base, '( true ( tried x 3 3 ) )'),
    (331, -1, AND, [435, 446, 579, 52], [458]),
    (332, -1, OR, [33, 287, 175], [520]),
    (333, -1, PROPOSITION, [226], [552, 458, 590, 359], base, '( true ( cell 3 3 x ) )'),
    (334, -1, AND, [215, 535, 136], [178]),
    (335, -1, AND, [166, 151], [128]),
    (336, -1, TRANSITION, [36], [163]),
    (337, -1, PROPOSITION, [524], [10, 437], base, '( true ( tried o 2 1 ) )'),
    (338, -1, AND, [151, 445, 8], [31]),
    (339, -1, PROPOSITION, [111], [146], other, '( next ( tried x 3 3 ) )'),
    (340, -1, AND, [114, 187, 64], [523]),
    (341, -1, AND, [573, 584], [84]),
    (342, -1, NOT, [219], [157]),
    (343, -1, NOT, [398], [392]),
    (344, -1, PROPOSITION, [80], [], goal, '( goal random 0 )'),
    (345, -1, OR, [43, 139], [191]),
    (346, -1, NOT, [330], [559]),
    (347, -1, PROPOSITION, [192], [], legal, '( legal x ( mark 2 3 ) )'),
    (348, -1, PROPOSITION, [590], [492], other, '( column 3 x )'),
    (349, -1, OR, [472, 134], [322]),
    (350, -1, AND, [171, 234], [128]),
    (351, -1, OR, [87, 97], [420]),
    (352, -1, PROPOSITION, [242], [410], other, '( next ( tried x 1 1 ) )'),
    (353, -1, OR, [201, 377, 251, 90, 274, 417, 98], [601]),
    (354, -1, PROPOSITION, [80], [], legal, '( legal random ( tiebreak xSecondChance ) )'),
    (355, -1, PROPOSITION, [250], [316], other, '( next ( cell 3 3 b ) )'),
    (356, -1, OR, [486, 548, 384], [496]),
    (357, -1, AND, [561, 438], [482]),
    (358, -1, PROPOSITION, [75], [103], other, '( next ( cell 3 1 b ) )'),
    (359, -1, AND, [231, 333, 185], [385]),
    (360, -1, AND, [66, 433], [611]),
    (361, -1, OR, [382, 573, 462], [74]),
    (362, -1, AND, [281, 215, 435, 136], [326]),
    (363, -1, AND, [516, 234, 199], [430]),
    (364, -1, PROPOSITION, [491], [], legal, '( legal o ( mark 3 3 ) )'),
    (365, -1, AND, [446, 52], [488]),
    (366, -1, AND, [596, 273, 147], [598]),
    (367, -1, AND, [446, 199, 52], [458]),
    (368, -1, OR, [114, 373], [589]),
    (369, -1, AND, [115, 147], [512]),
    (370, -1, PROPOSITION, [452], [325, 555], base, '( true ( tried x 1 2 ) )'),
    (371, -1, OR, [271, 112, 583, 396, 81, 329], [221]),
    (372, -1, AND, [442, 438], [482]),
    (373, -1, PROPOSITION, [], [129, 456, 444, 288, 95, 7, 499, 409, 208, 411, 211, 61, 176, 158, 368, 314, 261], input, '( does o ( mark 3 2 ) )'),
    (374, -1, AND, [209, 231, 581], [426]),
    (375, -1, AND, [535, 446, 579], [82]),
    (376, -1, PROPOSITION, [325], [452], other, '( next ( tried x 1 2 ) )'),
    (377, -1, PROPOSITION, [609], [353, 259, 65], base, '( true ( cell 1 2 x ) )'),
    (378, -1, AND, [66, 433, 196, 304], [100]),
    (379, -1, AND, [496, 561], [482]),
    (380, -1, PROPOSITION, [530], [99, 12, 202, 319, 71], base, '( true ( cell 2 2 o ) )'),
    (381, -1, OR, [209, 441, 571, 266, 193, 469, 25], [20]),
    (382, -1, PROPOSITION, [525], [167, 116, 235, 584, 361, 200], other, '( line o )'),
    (383, -1, PROPOSITION, [604], [525], other, '( column 3 o )'),
    (384, -1, AND, [66, 304], [356]),
    (385, -1, OR, [283, 359], [473]),
    (386, -1, PROPOSITION, [512], [188], other, '( sees_tmp5 x )'),
    (387, -1, AND, [66, 433, 435, 304], [575]),
    (388, -1, AND, [606, 87, 506, 196], [99]),
    (389, -1, PROPOSITION, [314], [9, 172, 140, 612], other, '( sees_tmp9 3 )'),
    (390, -1, PROPOSITION, [536], [289], other, '( next ( tried x 3 1 ) )'),
    (391, -1, TRANSITION, [280], [255]),
    (392, -1, PROPOSITION, [343], [], legal, '( legal x ( mark 3 1 ) )'),
    (393, -1, AND, [557, 281, 215, 136], [326]),
    (394, -1, PROPOSITION, [582], [], other, '( next ( cell 3 3 xSecondChance ) )'),
    (395, -1, AND, [442, 474], [591]),
    (396, -1, AND, [171, 234, 516, 196], [371]),
    (397, -1, AND, [166, 151, 416], [522]),
    (398, -1, PROPOSITION, [289], [343, 536], base, '( true ( tried x 3 1 ) )'),
    (399, -1, OR, [87, 171, 472], [587]),
    (400, -1, AND, [115, 147], [121]),
    (401, -1, AND, [446, 579], [614]),
    (402, -1, PROPOSITION, [470], [525], other, '( column 1 o )'),
    (403, -1, PROPOSITION, [268], [], other, '( next ( cell 2 1 xSecondChance ) )'),
    (404, -1, AND, [557, 171, 234, 516], [13]),
    (405, -1, AND, [494, 134], [512]),
    (406, -1, PROPOSITION, [180], [172, 241], other, '( sees_tmp1 1 x )'),
    (407, -1, AND, [557, 446, 579, 52], [458]),
    (408, -1, OR, [208, 69, 61, 158, 409, 261], [41]),
    (409, -1, AND, [214, 187, 373], [408]),
    (410, -1, TRANSITION, [352], [108]),
    (411, -1, OR, [87, 596, 373], [199]),
    (412, -1, AND, [209, 185, 475], [513]),
    (413, -1, OR, [43, 599], [487]),
    (414, -1, PROPOSITION, [477], [1, 213, 203, 93], other, '( sees_tmp11 1 )'),
    (415, -1, OR, [43, 141], [549]),
    (416, -1, PROPOSITION, [220], [605, 418, 397, 239, 329, 463], other, '( next_tmp7 2 )'),
    (417, -1, AND, [115, 540, 147], [353]),
    (418, -1, AND, [416, 446, 579], [82]),
    (419, -1, TRANSITION, [41], [69]),
    (420, -1, PROPOSITION, [351], [30], other, '( next ( tried o 2 2 ) )'),
    (421, -1, OR, [166, 596, 433], [64]),
    (422, -1, TRANSITION, [156], [581]),
    (423, -1, NOT, [97], [131]),
    (424, -1, AND, [557, 171, 234, 516], [430]),
    (425, -1, TRANSITION, [313], [187]),
    (426, -1, PROPOSITION, [374], [492], other, '( row 2 x )'),
    (427, -1, PROPOSITION, [284], [525], other, '( row 1 o )'),
    (428, -1, TRANSITION, [103], [136]),
    (429, -1, PROPOSITION, [10], [], legal, '( legal o ( mark 2 1 ) )'),
    (430, -1, OR, [448, 581, 363, 443, 424, 142, 323], [156]),
    (431, -1, TRANSITION, [247], [185]),
    (432, -1, PROPOSITION, [2], [104], other, '( marked 2 2 )'),
    (433, -1, PROPOSITION, [], [514, 421, 378, 497, 563, 519, 360, 49, 463, 33, 387, 272, 521, 15, 275, 119, 126], input, '( does o ( mark 1 3 ) )'),
    (434, -1, TRANSITION, [541], [327]),
    (435, -1, PROPOSITION, [], [387, 129, 441, 362, 331, 142, 188, 308, 554, 98], input, '( does random ( tiebreak x ) )'),
    (436, -1, AND, [166, 151], [230]),
    (437, -1, OR, [472, 337], [125]),
    (438, -1, PROPOSITION, [218], [357, 372], other, '( sees_tmp3 1 x )'),
    (439, -1, AND, [596, 321, 147], [598]),
    (440, -1, AND, [151, 445, 585], [31]),
    (441, -1, AND, [494, 435, 472, 134], [381]),
    (442, -1, PROPOSITION, [497], [44, 395, 511, 372], other, '( sees_tmp11 3 )'),
    (443, -1, AND, [540, 516, 234], [430]),
    (444, -1, OR, [219, 373], [60]),
    (445, -1, PROPOSITION, [], [457, 109, 566, 477, 338, 168, 466, 467, 256, 155, 62, 119, 38, 123, 554, 242, 85, 440], input, '( does x ( mark 1 1 ) )'),
    (446, -1, PROPOSITION, [478], [262, 545, 375, 418, 249, 582, 331, 495, 250, 264, 407, 548, 476, 602, 253, 56, 17, 365, 367, 401, 529], base, '( true ( cell 3 3 b ) )'),
    (447, -1, NOT, [547], [324]),
    (448, -1, AND, [516, 234, 585], [430]),
    (449, -1, AND, [115, 596, 196, 147], [598]),
    (450, -1, AND, [89, 151], [83]),
    (451, -1, AND, [234, 516], [217]),
    (452, -1, TRANSITION, [376], [370]),
    (453, -1, TRANSITION, [260], [24]),
    (454, -1, OR, [87, 171, 506, 516, 472, 134], [225]),
    (455, -1, PROPOSITION, [182], [212], other, '( next ( tried x 1 3 ) )'),
    (456, -1, AND, [114, 557, 187, 373], [537]),
    (457, -1, AND, [587, 151, 445], [31]),
    (458, -1, OR, [495, 331, 333, 367, 407, 476, 529], [610]),
    (459, -1, AND, [573, 584], [282]),
    (460, -1, PROPOSITION, [217], [9, 92], other, '( sees_tmp1 2 x )'),
    (461, -1, PROPOSITION, [26], [], legal, '( legal x ( mark 2 2 ) )'),
    (462, -1, NOT, [574], [67, 485, 361]),
    (463, -1, AND, [66, 433, 416], [100]),
    (464, -1, PROPOSITION, [259], [492], other, '( column 2 x )'),
    (465, -1, TRANSITION, [113], [151]),
    (466, -1, AND, [151, 445], [180]),
    (467, -1, AND, [151, 445], [488]),
    (468, -1, PROPOSITION, [210], [], legal, '( legal x ( mark 2 1 ) )'),
    (469, -1, AND, [494, 557, 472, 134], [381]),
    (470, -1, AND, [613, 24, 578], [402]),
    (471, -1, PROPOSITION, [195], [], legal, '( legal x ( mark 1 3 ) )'),
    (472, -1, PROPOSITION, [], [349, 159, 454, 441, 248, 47, 130, 132, 48, 268, 477, 437, 542, 237, 399, 469, 239], input, '( does o ( mark 2 1 ) )'),
    (473, -1, PROPOSITION, [385], [492], other, '( diagonal x )'),
    (474, -1, PROPOSITION, [230], [286, 395], other, '( sees_tmp3 1 o )'),
    (475, -1, PROPOSITION, [152], [412, 326, 552, 283], base, '( true ( cell 3 1 x ) )'),
    (476, -1, AND, [446, 64, 52], [458]),
    (477, -1, OR, [166, 281, 215, 472, 134, 445], [414]),
    (478, -1, TRANSITION, [316], [446]),
    (479, -1, OR, [166, 315], [484]),
    (480, -1, NOT, [118], [73]),
    (481, -1, AND, [606, 506], [217]),
    (482, -1, OR, [44, 46, 379, 203, 92, 188, 357, 96, 9, 170, 172, 213, 241, 372], [539]),
    (483, -1, AND, [66, 304], [488]),
    (484, -1, PROPOSITION, [479], [493], other, '( next ( tried o 1 1 ) )'),
    (485, -1, AND, [223, 462, 584], [577]),
    (486, -1, AND, [234, 516], [356]),
    (487, -1, TRANSITION, [413], [147]),
    (488, -1, OR, [467, 184, 531, 483, 285, 365, 120, 240, 244], [303]),
    (489, -1, PROPOSITION, [275], [553], other, '( marked 1 3 )'),
    (490, -1, NOT, [600], [224]),
    (491, -1, NOT, [255], [364]),
    (492, -1, OR, [426, 464, 513, 232, 473, 269, 348], [573]),
    (493, -1, TRANSITION, [484], [315]),
    (494, -1, PROPOSITION, [191], [159, 441, 248, 130, 405, 132, 72, 266, 48, 268, 17, 18, 328, 571, 193, 542, 237, 469, 25, 239, 244], base, '( true ( cell 2 1 b ) )'),
    (495, -1, AND, [587, 446, 52], [458]),
    (496, -1, PROPOSITION, [356], [213, 379], other, '( sees_tmp3 3 x )'),
    (497, -1, OR, [171, 433, 516, 304, 579, 52], [442]),
    (498, -1, PROPOSITION, [5], [102], other, '( next ( tried x 2 3 ) )'),
    (499, -1, AND, [187, 373], [128]),
    (500, -1, AND, [66, 304], [180]),
    (501, -1, AND, [87, 606, 214], [99]),
    (502, -1, AND, [114, 187], [121]),
    (503, -1, PROPOSITION, [37], [], legal, '( legal o ( mark 1 1 ) )'),
    (504, -1, AND, [596, 147], [128]),
    (505, -1, AND, [587, 281, 136], [326]),
    (506, -1, PROPOSITION, [], [220, 454, 161, 595, 149, 2, 531, 4, 517, 308, 95, 137, 206, 388, 481, 312, 528, 593], input, '( does x ( mark 2 2 ) )'),
    (507, -1, PROPOSITION, [80], [], legal, '( legal random ( tiebreak x ) )'),
    (508, -1, TRANSITION, [70], [293]),
    (509, -1, PROPOSITION, [551], [523, 552, 259], base, '( true ( cell 3 2 x ) )'),
    (510, -1, TRANSITION, [57], [189]),
    (511, -1, AND, [442, 150], [591]),
    (512, -1, OR, [545, 297, 299, 405, 312, 38, 278, 369, 165], [386]),
    (513, -1, PROPOSITION, [412], [492], other, '( column 1 x )'),
    (514, -1, AND, [66, 557, 433, 304], [570]),
    (515, -1, PROPOSITION, [257], [228], other, '( marked 3 1 )'),
    (516, -1, PROPOSITION, [], [160, 454, 497, 404, 443, 424, 5, 323, 227, 137, 448, 299, 363, 451, 120, 396, 142, 486], input, '( does x ( mark 2 3 ) )'),
    (517, -1, AND, [606, 506], [121]),
    (518, -1, AND, [215, 136], [230]),
    (519, -1, OR, [433, 189], [57]),
    (520, -1, PROPOSITION, [332], [140, 538], other, '( sees_tmp1 1 o )'),
    (521, -1, AND, [66, 557, 433, 304], [575]),
    (522, -1, OR, [169, 397, 194, 24, 85, 603], [260]),
    (523, -1, OR, [138, 129, 154, 340, 88, 509, 288], [122]),
    (524, -1, TRANSITION, [125], [337]),
    (525, -1, OR, [427, 222, 63, 383, 402, 177, 592], [382]),
    (526, -1, AND, [606, 104], [141]),
    (527, -1, OR, [43, 277], [148]),
    (528, -1, AND, [606, 506, 585], [267]),
    (529, -1, AND, [540, 446, 52], [458]),
    (530, -1, TRANSITION, [32], [380]),
    (531, -1, AND, [606, 506], [488]),
    (532, -1, PROPOSITION, [233], [271, 55, 48, 194, 126, 243], other, '( next_tmp5 3 )'),
    (533, -1, PROPOSITION, [144], [], other, '( next ( cell 1 2 xSecondChance ) )'),
    (534, -1, PROPOSITION, [611], [1, 291], other, '( sees_tmp3 3 o )'),
    (535, -1, PROPOSITION, [62], [375, 334, 61, 583, 237, 543], other, '( next_tmp5 1 )'),
    (536, -1, OR, [281, 398], [390]),
    (537, -1, PROPOSITION, [456], [], other, '( next ( cell 3 2 xSecondChance ) )'),
    (538, -1, AND, [520, 225], [591]),
    (539, -1, PROPOSITION, [482], [], sees, '( sees x ok )'),
    (540, -1, PROPOSITION, [47], [154, 161, 443, 417, 556, 529], other, '( next_tmp3 1 )'),
    (541, -1, PROPOSITION, [575], [434], other, '( next ( cell 1 3 x ) )'),
    (542, -1, AND, [494, 472], [128]),
    (543, -1, AND, [87, 606, 535], [99]),
    (544, -1, TRANSITION, [558], [296]),
    (545, -1, AND, [446, 52], [512]),
    (546, -1, PROPOSITION, [324], [91], other, '( next ( cell 2 3 b ) )'),
    (547, -1, PROPOSITION, [160], [447], other, '( marked 2 3 )'),
    (548, -1, AND, [446, 52], [356]),
    (549, -1, TRANSITION, [415], [606]),
    (550, -1, PROPOSITION, [102], [192, 5], base, '( true ( tried x 2 3 ) )'),
    (551, -1, TRANSITION, [122], [509]),
    (552, -1, AND, [333, 509, 475], [269]),
    (553, -1, NOT, [489], [124]),
    (554, -1, AND, [166, 435, 151, 445], [31]),
    (555, -1, NOT, [370], [16]),
    (556, -1, AND, [66, 540, 304], [575]),
    (557, -1, PROPOSITION, [], [514, 46, 582, 456, 404, 149, 90, 4, 424, 407, 288, 268, 566, 521, 155, 393, 576, 144, 469], input, '( does random ( tiebreak xSecondChance ) )'),
    (558, -1, PROPOSITION, [593], [544], other, '( next ( tried x 2 2 ) )'),
    (559, -1, PROPOSITION, [346], [], legal, '( legal x ( mark 3 3 ) )'),
    (560, -1, OR, [311, 596], [567]),
    (561, -1, PROPOSITION, [95], [379, 286, 291, 357], other, '( sees_tmp11 2 )'),
    (562, -1, AND, [281, 136], [236]),
    (563, -1, AND, [66, 433, 273], [100]),
    (564, -1, AND, [281, 136, 64], [326]),
    (565, -1, PROPOSITION, [212], [182, 195], base, '( true ( tried x 1 3 ) )'),
    (566, -1, AND, [166, 557, 151, 445], [31]),
    (567, -1, PROPOSITION, [560], [279], other, '( next ( tried o 1 2 ) )'),
    (568, -1, PROPOSITION, [326], [152], other, '( next ( cell 3 1 x ) )'),
    (569, -1, PROPOSITION, [121], [44, 203], other, '( sees_tmp3 2 x )'),
    (570, -1, PROPOSITION, [514], [], other, '( next ( cell 1 3 xSecondChance ) )'),
    (571, -1, AND, [494, 134, 199], [381]),
    (572, -1, PROPOSITION, [174], [], legal, '( legal x ( mark 3 2 ) )'),
    (573, -1, PROPOSITION, [492], [341, 167, 116, 223, 459, 361], other, '( line x )'),
    (574, -1, PROPOSITION, [17], [462], other, 'open'),
    (575, -1, OR, [387, 521, 327, 105, 3, 6, 556], [541]),
    (576, -1, AND, [557, 281, 215, 136], [162]),
    (577, -1, OR, [116, 485], [252]),
    (578, -1, PROPOSITION, [77], [580, 12, 178, 470], base, '( true ( cell 3 1 o ) )'),
    (579, -1, PROPOSITION, [], [262, 375, 418, 249, 582, 331, 497, 264, 407, 602, 54, 253, 11, 15, 176, 401, 314], input, '( does o ( mark 3 3 ) )'),
    (580, -1, AND, [163, 69, 578], [63]),
    (581, -1, PROPOSITION, [422], [430, 590, 374], base, '( true ( cell 2 3 x ) )'),
    (582, -1, AND, [557, 446, 579, 52], [394]),
    (583, -1, AND, [171, 234, 535], [371]),
    (584, -1, NOT, [382], [341, 67, 485, 459]),
    (585, -1, PROPOSITION, [176], [448, 105, 274, 266, 440, 528], other, '( next_tmp1 3 )'),
    (586, -1, AND, [197, 229], [591]),
    (587, -1, PROPOSITION, [399], [138, 505, 495, 251, 457, 6], other, '( next_tmp1 2 )'),
    (588, -1, PROPOSITION, [320], [], legal, '( legal o ( mark 1 3 ) )'),
    (589, -1, PROPOSITION, [368], [145], other, '( marked 3 2 )'),
    (590, -1, AND, [581, 327, 333], [348]),
    (591, -1, OR, [1, 286, 291, 93, 110, 538, 34, 586, 301, 140, 612, 395, 511], [39]),
    (592, -1, PROPOSITION, [202], [525], other, '( row 2 o )'),
    (593, -1, OR, [296, 506], [558]),
    (594, -1, TRANSITION, [221], [112]),
    (595, -1, AND, [606, 506, 8], [267]),
    (596, -1, PROPOSITION, [], [560, 421, 90, 265, 95, 51, 98, 504, 449, 55, 411, 59, 366, 119, 175, 144, 439], input, '( does o ( mark 1 2 ) )'),
    (597, -1, PROPOSITION, [78], [68], other, '( next ( tried x 2 1 ) )'),
    (598, -1, OR, [55, 449, 366, 439, 51, 293], [70]),
    (599, -1, PROPOSITION, [205], [413], other, '( next ( cell 1 2 b ) )'),
    (600, -1, PROPOSITION, [53], [19, 490], base, '( true ( tried o 3 1 ) )'),
    (601, -1, PROPOSITION, [353], [609], other, '( next ( cell 1 2 x ) )'),
    (602, -1, AND, [273, 446, 579], [82]),
    (603, -1, AND, [166, 214, 151], [522]),
    (604, -1, AND, [28, 112, 163], [383]),
    (605, -1, AND, [215, 416, 136], [178]),
    (606, -1, PROPOSITION, [549], [179, 27, 161, 595, 149, 531, 4, 517, 308, 501, 206, 388, 17, 481, 312, 143, 526, 23, 543, 528, 243], base, '( true ( cell 2 2 b ) )'),
    (607, -1, AND, [215, 273, 136], [178]),
    (608, -1, TRANSITION, [306], [231]),
    (609, -1, TRANSITION, [601], [377]),
    (610, -1, PROPOSITION, [458], [226], other, '( next ( cell 3 3 x ) )'),
    (611, -1, OR, [249, 21, 360], [534]),
    (612, -1, AND, [389, 229], [591]),
    (613, -1, PROPOSITION, [29], [202, 263, 470], base, '( true ( cell 2 1 o ) )'),
    (614, -1, OR, [298, 401, 7], [22]),
)

# DONE
