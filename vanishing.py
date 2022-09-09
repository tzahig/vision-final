"""
Manually add vanishing lines and vanishing points.
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

GROUPS_DICT = {
    'entrance/yossi_door1.jpg': [
        [(496, 427), (919, 435), (523, 1319), (897, 1304)],
        [(732, 1458), (757, 1606), (984, 1447), (1054, 1590)],
        [(496, 427), (523, 1319), (919, 435), (897, 1304)],
    ],
    'entrance/yossi_door2.jpg': [
        [(687, 384), (1117, 376), (692, 1257), (1066, 1286)],
        [(831, 1415), (789, 1559), (1091, 1439), (1097, 1592)],
        [(687, 384), (692, 1257), (1117, 376), (1066, 1286)],
    ],
    'entrance/tzahi_door1.jpg': [
        [(4 * 231, 4 * 659), (4 * 453, 4 * 590), (4 * 205, 4 * 40), (4 * 448, 4 * 70)],
        [(4 * 479, 4 * 119), (4 * 483, 4 * 517), (4 * 838, 4 * 106), (4 * 803, 4 * 482)],
        [(4 * 353, 4 * 660), (4 * 410, 4 * 737), (4 * 468, 4 * 622), (4 * 537, 4 * 686)],
    ],
    'entrance/tzahi_door2.jpg': [
        [(4 * 304, 4 * 75), (4 * 525, 4 * 84), (4 * 324, 4 * 638), (4 * 512, 4 * 608)],
        [(4 * 565, 4 * 114), (4 * 568, 4 * 513), (4 * 816, 4 * 70), (4 * 786, 4 * 590)],
        [(4 * 431, 4 * 746), (4 * 411, 4 * 663), (4 * 576, 4 * 718), (4 * 533, 4 * 640)],
    ],
    # 'data/images/dslr_images_undistorted/DSC_1000.JPG': [
    #     [(567, 1181), (2024, 1383), (519, 1801), (2002, 1902)],
    #     [(567, 1181), (519, 1801), (2024, 1383), (2002, 1902)],
    #     [(216, 1000), (441, 1224), (181, 1460), (484, 1634)],
    #
    # ],
    'data/observatory/images/dslr_images_undistorted/DSC_1000.JPG': [
        [(1235, 4125), (1275, 3447), (2847, 4129), (2870, 3360)],
        # [(3067, 3273), (3038, 4102), (4669, 3054), (4662, 4123)],
        [(3154, 3405), (4485, 3212), (3135, 4042), (4476, 3723)],
        [(680, 3444), (734, 3351), (4377, 3035), (4281, 3008)],
    ],
    'data/observatory/images/dslr_images_undistorted/DSC_0996.JPG': [
        [(1177, 4103), (1229, 3527), (2802, 4027), (2927, 3319)],
        [(2935, 3463), (4162, 3319), (2864, 3946), (4071, 3724)],
        [(707, 3476), (818, 3141), (4106, 3091), (3953, 2975)],
    ],
    'data/living_room/images/dslr_images_undistorted/DSC_1586.JPG': [
        [(609, 575), (1349, 3954), (3160, 313), (3117, 2626)],
        [(3380, 3495), (3951, 2914), (3983, 3839), (4462, 3124)],
        [(3380, 3495), (3983, 3839), (3951, 2914), (4462, 3124)],
    ],
    'data/living_room/images/dslr_images_undistorted/DSC_1587.JPG': [
        [(424, 130), (1209, 3584), (5, 1727), (737, 4106)],
        # [(1128, 72), (1702, 3571), (3491, 325), (3383, 2550)],
        [(2690, 4118), (4325, 2800), (3811, 4141), (4847, 2978)],
        [(4204, 2902), (4705, 3130), (2898, 3269), (3339, 3587)],
    ],
    # [
    # [(1162, 290), (1751, 3881), (3493, 281), (3393, 2342)],
    # [(3584, 3397), (4201, 2900), (4149, 3761), (4709, 3131)],
    # [(3584, 3397), (4149, 3761), (4201, 2900), (4709, 3131)],
}
COLORS = ['r', 'g', 'b']

POINT_INDEX = 0
LAST_COORD = None


def view_vanishing(image1, image2, points1, points2):
    from matplotlib import pyplot as plt

    points = points1
    plt.imshow(image1, cmap='gray')
    for point in points:
        plt.scatter(*point)
    plt.show()


def get_vanishing(fn):
    return np.vstack(vanishing_points(fn))


def picker(img):
    img = imread(img)
    ax = plt.gca()
    fig = plt.gcf()
    ax.imshow(img)

    # each group contains 4 pts defining 2 lines [line11, line12, line21, line22]
    groups = [[], [], []]

    def save(coord):
        global POINT_INDEX
        group_index = POINT_INDEX // 4
        point_number = POINT_INDEX % 2
        c = COLORS[group_index]
        ax.scatter(*coord, c=c)
        fig.canvas.draw_idle()
        groups[group_index].append(coord)
        print(f'{POINT_INDEX} : {group_index}, {point_number} {coord}')
        print(groups)
        POINT_INDEX += 1

    def onclick(event):
        global LAST_COORD
        if event.xdata != None and event.ydata != None:
            coord = (event.xdata, event.ydata)
            coord = tuple([int(round(x)) for x in coord])
            print(coord)
            LAST_COORD = coord

    def onpress(event):
        k = event.key
        print(k)
        if k == 'x':
            save(LAST_COORD)

    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
    plt.show()


def intersection(a1, a2, b1, b2):
    """
    Return the intersection point  of two lines
    defined by (a1,a2) and (b1,b2).
    """
    h = np.hstack([[a1, a2, b1, b2], np.ones((4, 1))])
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    assert z != 0, 'lines do not intersect'
    return int(x / z), int(y / z)


def draw_lines(fn, draw_points=False, out_path=None):
    groups = GROUPS_DICT[fn]
    img = imread(fn)
    plt.imshow(img)
    for group_index, pts in enumerate(groups):
        if len(pts) != 4:
            continue
        line1 = pts[0], pts[1]
        line2 = pts[2], pts[3]
        c = COLORS[group_index]

        def get_xy(line):
            xs = line[0][0], line[1][0]
            ys = line[0][1], line[1][1]
            return xs, ys

        plt.plot(*get_xy(line1), c=c)
        plt.plot(*get_xy(line2), c=c)

        if draw_points:
            if group_index in [0, 1, 2]:
                v = intersection(*pts)
                plt.scatter(*v, c=c)

    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()

    plt.clf()


def vanishing_points(fn):
    groups = GROUPS_DICT[fn]

    assert len(groups) == 3
    for pts in groups:
        assert len(pts) == 4

    return [intersection(*pts) for pts in groups]


def main():
    # fn = 'entrance/yossi_door1.jpg'
    # fn = 'data/observatory/images/dslr_images_undistorted/DSC_1000.JPG'
    fn = 'data/observatory/images/dslr_images_undistorted/DSC_0996.JPG'
    # fn = 'data/living_room/images/dslr_images_undistorted/DSC_1586.JPG'
    # fn = 'data/living_room/images/dslr_images_undistorted/DSC_1587.JPG'

    mode = 'picker'
    mode = 'display'

    if mode == 'picker':
        picker(fn)
    elif mode == 'display':
        draw_lines(fn, draw_points=False)
    else:
        raise ValueError(f'Invalid mode {mode}')


if __name__ == '__main__':
    main()
