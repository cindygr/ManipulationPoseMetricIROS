#!/usr/bin/env python3

import numpy as np
from scipy import spatial
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class PoseDistance:
    def __init__(self):
        # Assumes unit sphere
        self.mass_matrix = np.identity(6)

        # for finding translations
        self.pt_zero = np.zeros([4,1])
        self. pt_zero[3] = 1

        # for finding rotation (3 vectors)
        self.mat_rot_only = np.identity(4)
        self.mat_rot_only[3,3] = 0

    def set_mass_2D(self, span, depth):
        self.mass_matrix[0,0] = 1/span
        self.mass_matrix[1,1] = 1/depth
        self.mass_matrix[2,2] = 2/np.pi

    def set_mass_3D_sphere(self, radius ):
        self.mass_matrix[0,0] = 1/radius
        self.mass_matrix[1,1] = 1/radius
        self.mass_matrix[2,2] = 2/np.pi

    def get_translation(self, m1, m2):

        pt_center1 = m1 @ self.pt_zero
        pt_center2 = m2 @ self.pt_zero

        return pt_center2 - pt_center1

    def get_rotation(self, m1, m2):
        m1_rot = m1 @ self.mat_rot_only
        m2_rot = m2 @ self.mat_rot_only

        # Make this a rotation from the origin to somewhere
        m_rot = m2_rot[0:3,0:3] @ m1_rot[0:3,0:3].transpose()
        m_rot = R.from_dcm( m_rot[0:3,0:3] )

        # Extract as a quaternion
        self.q = m_rot.as_quat()

        return self.q

    def get_time_derivative_body_frame(self, m1, m2):
        m_body = np.linalg.inv(m1) @ m2
        m_identity = np.identity(4)
        d_trans = self.get_translation(m_identity, m_body)
        d_quat = self.get_rotation(m_identity, m_body)
        d_quat_R = R.from_quat(d_quat)
        d_euc = d_quat_R.as_euler('xyz', degrees=False)

        return [d_trans[0,0], d_trans[1,0], d_trans[2,0], d_euc[0], d_euc[1], d_euc[2]]

    def distance_RT(self, m1, m2 ):

        t = self.get_translation(m1, m2)
        q = self.get_rotation(m1, m2)

        # Euclidean distance of translation plus quaternion distance of rotation (angle rotated)
        dist_T = spatial.distance.euclidean( [0,0,0], t[0:3] )
        dist_R = 2.0 * np.arctan2( spatial.distance.euclidean(q[0:3], [0,0,0]), q[3] )

        return dist_T + dist_R

    def distance_LI(self, m1, m2 ):
        return 0

    def distance_RT_seq(self, ms):
        if len(ms) < 2:
            raise ValueError('ms needs to have at least two matrices\n')

        d_sum = 0
        for i in range(0,len(ms)-1):
            time_deriv = self.get_time_derivative_body_frame(ms[i], ms[i+1])
            d_step_cost = time_deriv @ self.mass_matrix @ np.transpose(time_deriv)
            d_sum += d_step_cost

        return d_sum

    def get_sequence_RT(self, m1, m2, n = 10):

        if n < 2:
            n = 2

        # Translation/rotation to m1
        t_m1 = self.get_translation(np.identity(4), m1)
        q_m1 = self.get_rotation(np.identity(4), m1)

        # Translation/rotation to m2
        t_m2 = self.get_translation(np.identity(4), m2)
        q_m2 = self.get_rotation(np.identity(4), m2)

        # For rotation interpolation
        r1r2 = R.from_quat([q_m1, q_m2])
        slerp = Slerp([0,1], r1r2)

        # Our list of matrices to return
        ms = []

        # identity transforms/rotates - these will be filled in during loop
        m_trans = np.identity(4)
        m_rot = np.identity(4)

        # linearly interpolate between 0 and 1
        qs = slerp(np.linspace(0,1, n))
        for i,dt in enumerate( np.linspace(0,1, n) ):
            # move a bit
            trans = (1-dt) * t_m1 + dt * t_m2
            m_trans[0:3, 3] = np.transpose( trans[0:3] )

            # Rotate by desired amount
            m_rot[0:3,0:3] = qs[i].as_dcm()

            # Move to m1
            m_add = m_trans @ m_rot
            ms.append( m_add )

        return ms


def write_pt(m):
    pt_zero = np.zeros([4, 1])
    pt_zero[3] = 1

    print( np.transpose( m @ pt_zero ) )
    for i in range(0,3):
        vec_zero = np.zeros([4,1])
        vec_zero[i] = 1
        print( np.transpose(m @ vec_zero) )

    print('\n')

if __name__ == '__main__':
    print('Checking Pose Distance Metric code\n')

    m1 = np.identity(4)
    d_ang = np.pi / 3
    m1[0:3,3] = [1,1,1]
    m1[0:3,0:3] = R.as_dcm( R.from_euler( 'XYZ', [0,0,d_ang]) )


    m2 = np.identity(4)
    m2[0:3,3] = [2,2,2]

    pd = PoseDistance()

    dist_RT = pd.distance_RT(m1, m2)
    dist_RT_seq = pd.distance_RT_seq( pd.get_sequence_RT(m1, m2, n = 10) )
    print('Distance RT, should be {0:0.2f}, is {1:0.2f}, integrated {2:0.2f}\n'.format( np.sqrt(3) + d_ang, dist_RT, dist_RT_seq ) )

    mat_seq2 = pd.get_sequence_RT(m1, m2, 2)

    print('M1:')
    write_pt(m1)
    print('M2:')
    write_pt(m2)

    for i,m in enumerate(mat_seq2):
        print('M{}\n'.format(i))
        write_pt(m)

    mat_seq4 = pd.get_sequence_RT(m1, m2, 8)

    for i,m in enumerate(mat_seq4):
        print('M{}\n'.format(i))
        write_pt(m)

