import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,voronoi_plot_2d



class ViewSegment(object):
    def __init__(self):
        pass

    def get_voronoi(self, points):
        vor = Voronoi(points)
        return vor

    def get_segments(self, vor):
        # from :  https://github.com/scipy/scipy/blob/master/scipy/spatial/_plotutils.py
        center = vor.points.mean(axis=0)
        # ptp_bound = vor.points.ptp(axis=0) # range of values max to min
        # ptp_bound = ptp_bound.max()
        ptp_bound = 140
        segments = dict() # following vertices index
        finite_segments = []
        infinite_segments = []
        far_points = []
        far_point_count = 0
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            seg_point = []
            seg_point_idx = []
            infinite_point = False
            if np.all(simplex >= 0):
                finite_segments.append(vor.vertices[simplex])
                seg_point = vor.vertices[simplex]
                seg_point_idx = list(simplex)
            else:
                infinite_point = True
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if (vor.furthest_site):
                    direction = -direction
                far_point = vor.vertices[i] + direction * ptp_bound

                infinite_segments.append([vor.vertices[i], far_point])

                seg_point = [vor.vertices[i], far_point]
                far_points.append(far_point)
                far_point_count += 1
                seg_point_idx = [i, -far_point_count] # assume far points are unique

            # append the seg point to each segments intelligently
            for idx in range(2):
                if pointidx[idx] not in segments:
                    segments[pointidx[idx]] = []
                segments[pointidx[idx]].append(seg_point_idx) # add lines to each region, rearrange later


        # print(segments)
        # rearrange segments to connect lines
        for seg_id in segments:
            list_of_lines = segments[seg_id]
            DAG = []
            DAG.extend(list_of_lines[0])

            # print(list_of_lines)

            for x in list_of_lines:
                # print(x)
                [a,b] = x
                if a in DAG and b in DAG:
                    continue
                if DAG[-1] == a:
                    DAG.append(b)
                elif DAG[-1] == b:
                    DAG.append(a)
                elif DAG[0] == a:
                    DAG.insert(0, b)
                elif DAG[0] == b:
                    DAG.insert(0, a)
                else:
                    list_of_lines.append([a,b]) # process later
                    # print(f"Appending {[a,b]}, given DAG {DAG}")

            # print(list_of_lines)
            # print(DAG)
            segments[seg_id] = DAG

        # print(segments)


        # replace seg idx with actual points
        for seg_id in segments:
            actual_seg= []
            for seg_point_id in segments[seg_id]:
                # print(seg_point_id)
                if seg_point_id < 0:
                    actual_seg.append(list(far_points[abs(seg_point_id)-1]))
                else:
                    actual_seg.append(list(vor.vertices[seg_point_id]))
            segments[seg_id] = actual_seg
        # print(segments)
        return segments

def test_voronoi():
    vs = ViewSegment()
    points = np.array([[0, 0], [0, 2], [1, 0], [1, 1], [2,2]])
    vor = vs.get_voronoi(points)
    print(vor.points)
    print(vor.vertices)
    print(vor.ridge_points)
    print(vor.ridge_vertices)
    print(vor.regions)
    print(vor.point_region)
    voronoi_plot_2d(vor)
    plt.savefig('tmp.png')
    segments = vs.get_segments(vor)
    print(segments)



def numpy_to_ply(path, output, format='npy'):
    if format == 'npy':
        xyz = np.load(path)
    elif format == 'bin':
        xyz = np.fromfile(path, dtype=np.float32)
        xyz = np.reshape(xyz, (-1, 4)) # xyz intensity
        xyz = xyz[:, :3]
    print(xyz.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(output, pcd)

def o3d_test():
    test_data = "test_data/"
    print(o3d)
    # test_ply = "fragment.ply"
    test_ply = test_data + "test.ply"
    test_np = test_data + "test.npy"
    test_np = test_data + "test2.npy"
    numpy_to_ply(test_np, test_ply, format='npy')
    # test_bin = test_data + "test.bin"
    # numpy_to_ply(test_bin, test_ply, format='bin')

    # test_ply = "test_trace.ply"
    pcd = o3d.io.read_point_cloud(test_ply)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    #crop
    crop_json = "crop_test.json"
    # crop_json = "trace_crop_test.json"
    # crop_json = "crop.json"
    vol = o3d.visualization.read_selection_polygon_volume(
        crop_json)
    cropped_pcd = vol.crop_point_cloud(pcd)
    print(cropped_pcd)
    print(np.asarray(cropped_pcd.points))
    o3d.visualization.draw_geometries([cropped_pcd])

def o3d_viewer():
    test_ply = "test_trace.ply"
    # test_ply = "test_trace_crop.ply"

    pcd = o3d.io.read_point_cloud(test_ply)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # test_voronoi()
    o3d_test()
    # o3d_viewer()