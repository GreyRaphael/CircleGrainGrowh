import numpy as np
import pylab as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image, ImageDraw


def generate_2_gains(nx, ny):
    x, y = np.ogrid[:nx + 1, :ny + 1]

    radius = nx / 3
    etas = np.zeros((2, nx + 1, ny + 1))
    etas[0, :, :] = 1
    etas[0, :, :][(x - nx / 2) ** 2 + (y - ny / 2) ** 2 < radius ** 2] = 0.0
    etas[1, :, :][(x - nx / 2) ** 2 + (y - ny / 2) ** 2 < radius ** 2] = 1.0

    return etas.reshape(2, -1, 1)


def generate_voronoi(points_number, Nx, Ny):
    # dx, dy = 0.5, 0.5
    points_sections = []

    # center section for generating symmetry points
    points_xrange = np.random.randint(0, Nx + 1, size=(points_number, 1))
    points_yrange = np.random.randint(0, Ny + 1, size=(points_number, 1))
    points_sections0 = np.hstack((points_xrange, points_yrange))

    points_sections.append(points_sections0)
    for i in range(1, 9):
        points_sections.append(np.copy(points_sections[0]))

    # assign 9 sections
    points_sections[1][:, 1] -= Ny
    points_sections[2][:, 1] -= Ny
    points_sections[2][:, 0] += Nx
    points_sections[3][:, 0] += Nx
    points_sections[4][:, 1] += Ny
    points_sections[4][:, 0] += Nx
    points_sections[5][:, 1] += Ny
    points_sections[6][:, 1] += Ny
    points_sections[6][:, 0] -= Nx
    points_sections[7][:, 0] -= Nx
    points_sections[8][:, 1] -= Ny
    points_sections[8][:, 0] -= Nx

    # stack 9 sections
    points = np.vstack(points_sections)
    vor = Voronoi(points)

    ## 通过ridge相邻的points
    outer_points = []
    for ridge_points in vor.ridge_dict:
        # ridge_points=(ridge_pointHead, ridge_pointEnd)
        if ridge_points[0] in range(points_number) or ridge_points[1] in range(points_number):
            outer_points.append(ridge_points)

    new_points = []
    for points in outer_points:
        if points[0] not in range(points_number) and points[0] not in new_points:
            new_points.append(points[0])
        if points[1] not in range(points_number) and points[1] not in new_points:
            new_points.append(points[1])

    ## 通过vertice相邻的points
    outout_points = []
    for ridge_points in vor.ridge_dict:
        # ridge_points=(ridge_pointHead, ridge_pointEnd)
        verticeHeadIndex, verticeEndIndex = vor.ridge_dict[ridge_points]
        # -1这一句很重要，出现polygon bug
        if verticeHeadIndex != -1 and verticeEndIndex != -1:
            verticeHeadX, verticeHeadY = vor.vertices[verticeHeadIndex]
            verticeEndX, verticeEndY = vor.vertices[verticeEndIndex]
            if (0 < verticeHeadX < Nx and 0 < verticeHeadY < Ny) or (0 < verticeEndX < Nx and 0 < verticeEndY < Ny):
                outout_points.append(ridge_points)

    newnew_points = []
    for points in outout_points:
        if points[0] not in range(points_number) and points[0] not in newnew_points:
            newnew_points.append(points[0])
        if points[1] not in range(points_number) and points[1] not in newnew_points:
            newnew_points.append(points[1])

    # 将new_points, newnew_points汇集到allout_points
    allout_points = new_points.copy()
    for point_index in newnew_points:
        if point_index not in allout_points:
            allout_points.append(point_index)

    # 编号：有几个point就有几个region；一个grain可以对应多个region(周期性边界条件)
    totoal_polygon_number = points_number + len(allout_points)
    # point_polygons: [region_index, grain_index]
    point_polygons = np.zeros((totoal_polygon_number, 2), dtype='int')
    point_polygons[:points_number, 0] = vor.point_region[:points_number]
    point_polygons[:points_number, 1] = np.arange(points_number)
    point_polygons[points_number:totoal_polygon_number, 0] = vor.point_region[allout_points]
    point_polygons[points_number:totoal_polygon_number, 1] = np.array(
        [point_index % points_number for point_index in allout_points])

    # initialize order parameter
    order_matrix = np.zeros((points_number, Ny, Nx))
    for region_index, grain_index in point_polygons:
        # get polygon
        region = vor.regions[region_index]
        polygon = [tuple(vor.vertices[i]) for i in region]

        # draw mask
        img = Image.new('L', (Nx, Ny), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=0, fill=1)
        mask = np.array(img, dtype='bool')

        order_matrix[grain_index, :, :][mask] = 1

    return order_matrix


if __name__ == '__main__':
    # # test 2 grains
    # Etas = generate_2_gains(80, 80)
    # print(Etas.shape)
    # # imshow results
    # for i in range(2):
    #     fig = plt.figure()
    #     plt.imshow(Etas[i, :, :])
    # plt.show()

    ## test voronoi
    order_matrix = generate_voronoi(25, 100, 100)
    print(order_matrix.shape)
