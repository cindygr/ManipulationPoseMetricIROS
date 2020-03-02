#!/usr/bin/env python3

from PoseDistanceMetric import PoseDistance
import csv
import numpy as np
from scipy import spatial
from scipy.spatial.transform import Rotation as R



pd = PoseDistance()

def Process2DCSV(xyt, xyt_target):
    m1 = np.identity(4)
    m1[0:2,3] = xyt[0:2]
    m1[0:3,0:3] = R.as_dcm( R.from_euler('XYZ',[0,0,xyt[2]], degrees=True) )

    m2 = np.identity(4)
    m2[0:2,3] = xyt_target[0:2]
    m2[0:3,0:3] = R.as_dcm( R.from_euler('XYZ',[0,0,xyt_target[2]], degrees=True) )

    return pd.distance_RT(m1, m2)


def Convert6dofPoseToMatrix( pose ):
    m = np.identity(4)
    m[0:3,3] = pose[0:3]
    m[0:3,0:3] = R.as_dcm( R.from_quat( pose[3:7]) )
    return m


def Process3DCSV_final_pose_err(target, actual):
    m1 = Convert6dofPoseToMatrix( target )

    m2 = Convert6dofPoseToMatrix( actual )

    return pd.distance_RT(m1, m2)


def Process3DCSV_path_len_efficiency(target, final, pose_seq):
    m_target = Convert6dofPoseToMatrix( target )
    m_final = Convert6dofPoseToMatrix( final )

    n = 8
    dist_opt = pd.distance_RT_seq( pd.get_sequence_RT(m_target, m_final, n) )
    ms = []
    for p in pose_seq:
        ms.append( Convert6dofPoseToMatrix( p ) )

    dist_path = pd.distance_RT_seq(ms)
    return dist_path / dist_opt


def Process3DCSV_path_len_err(target, final, pose_seq):
    m_target = Convert6dofPoseToMatrix( target )
    m_final = Convert6dofPoseToMatrix( final )

    m_opt = pd.get_sequence_RT( m_target, m_final, 8 )
    dist_opt = pd.distance_RT_seq( m_opt )
    ms = []
    for p in pose_seq:
        ms.append( Convert6dofPoseToMatrix( p ) )

    while (len(ms) > 100):
        ms = ms[:2:]

    d_min = []
    for m_o in m_opt:
        d_closest = 1e30
        for m in ms:
            d = pd.distance_RT(m_o, m)
            if d < d_closest:
                d_closest = d
        d_min.append( d_closest )

    d_biggest = max( d_min )
    return d_biggest

def read_csv_2D():
    print('2D csv read\n')

    d_done = []
    strs = {"averaged","separated", "complete"}
    for s in strs:
        fname = 'DataIROS/' + s + '-handResults-fixed.csv'
        with open(fname) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:

                r = []
                str_all = ''
                for w in row:
                    print('{}, '.format(w), end='')
                    r.append(w)
                    str_all = str_all + w + ','
                print('\n')
                try:
                    xyt = list(map(float, row[1:4]))
                    xyt_target = list(map(float, row[4:7]))
                    pd.set_mass_2D( float(row[1] ), float(row[2] ))
                    d = Process2DCSV(xyt, xyt_target)
                    print('\t{} has distance {:0.2f}\n'.format( row[0], Process2DCSV(xyt, xyt_target) ) )
                    d_done.append(str_all + str(d) + '\n')
                except:
                    d_done.append(str_all + 'Differance\n')

        fname = 'DataIROS/' + s + '-handResults.csv'
        with open(fname, mode = 'w') as file:
            for d in d_done:
                file.write(d)


def read_and_convert_csv_3D(fname, sphere_rad = 1):
    print('3D csv read {}'.format(fname))

    full_fname = 'testpoint_data/' + fname + '.csv'
    ps = []
    try:
        with open(full_fname) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if len(row) == 7:
                    pose = list(map(float, row[0:7]))
                    for i in range(0,3):
                        pose[i] = pose[i] / sphere_rad
                    d_ang = pose[3]
                    pose[3:6] = pose[4:7]
                    pose[6] = d_ang
                else:
                    pose = list(map(float, row[0:3]))
                    for i in range(0,3):
                        pose[i] = pose[i] / sphere_rad
                    pose[3:7] = [0,0,0,1]

                ps.append(pose)
    except IOError:
       print( "Error: File {} does not appear to exist.".format(fname) )

    return ps


def read_csv_3D(fname, p_home, p_target, sphere_rad):
    ps = read_and_convert_csv_3D(fname, sphere_rad)

    if len(ps) < 2:
        return 'Nan, Nan, Nan, Nan'

    p_final = ps[-1]
    print('Final: {} Target: {}'.format(p_final[0:3], p_target[0:3]))
    # Use the orientation in home
    p_target[3:-1] = p_home[3:-1]
    d_home = Process3DCSV_final_pose_err(ps[0], p_home)
    d_target = Process3DCSV_final_pose_err(p_final, p_target)
    d_path_eff = Process3DCSV_path_len_efficiency(p_home, p_target, ps)
    d_frechet = Process3DCSV_path_len_err(p_home, p_target, ps)

    print('home {:0.4f} target {:0.4f} eff {:0.4f} Frechet {:0.4f}'.format(d_home, d_target, d_path_eff, d_frechet) )
    out_str = str(d_home) + ',' + str(d_target) + ',' + str(d_path_eff) + ',' + str(d_frechet)

    return out_str


def process_all_3D():
    print('3D csv read\n')

    sphere_radius = [0.85, 0.9849, 1, 0.9849]
    arms = ['UR5', 'Kinova7', 'WAM', 'Kinova4']
    #arms = ['Kinova7']
    #sphere_radius = [0.9849]
    strs = []
    for r,arm_name in zip( sphere_radius, arms ):
        #pd.set_mass_3D_sphere( r )
        for size in ['Large', 'Small']:
            fname_input = 'Offsets/{0}_Input_{1}_Home_Pos'.format(arm_name, size)
            pose_initial = read_and_convert_csv_3D(fname_input)

            fname_change = 'Offsets/{0}_{1}_Input_Offsets'.format(arm_name, size)
            pose_change = read_and_convert_csv_3D(fname_change)

            for i, hp in enumerate( range(1,8) ):
                for j, tp in enumerate( range(1,15) ):
                    fname = '{0}/{1}/Combined Data/{0}_{1}_Point_and_Orientation_HP{2}_TP{3}'.format(arm_name, size,hp,tp)
                    if arm_name == "UR5":
                        fname = '{0}/{1}/Combined Data/x11_{0}_{1}_Point_and_Orientation_HP{2}_TP{3}'.format(arm_name, size,hp,tp)
                    str_dist = read_csv_3D(fname, pose_initial[i], pose_change[j], r )
                    str_add = '{},{},HP{},TP{},'.format(arm_name, size, hp,tp) + str_dist + '\n'
                    strs.append( str_add )

        full_out_fname = "testpoint_data/dists" + arm_name + ".csv"
        with open(full_out_fname, mode = 'w') as file:
            for s in strs:
                file.write(s)


    full_out_fname = 'testpoint_data/dists.csv'
    with open(full_out_fname, mode = 'w') as file:
        for s in strs:
            file.write(s)


if __name__ == '__main__':
    read_csv_2D()
