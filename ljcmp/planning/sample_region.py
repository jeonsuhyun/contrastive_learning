
import time
import copy
import numpy as np

from srmt.constraints.constraints import ConstraintBase, ConstraintIKBase
from ljcmp.models.latent_model import LatentModel, LatentValidityModel

from queue import PriorityQueue
from itertools import count
import pyquaternion

class RegionSampler(object):
    def __init__(self, constraint:ConstraintIKBase) -> None:
        self.constraint = constraint
        self.pose = None

    def set_target_pose(self, pose):
        self.pose = pose

    def sample(self):
        assert self.pose is not None, 'set pose first'

        self.constraint.update_target(self.pose)
        while True:
            q = np.random.uniform(self.constraint.lb, self.constraint.ub)
            r = self.constraint.solve_ik(q, self.pose)
            # print('################ f=', constraint.function_ik(q), '######################')
            if r is False:
                continue
            
            # self.constraint.planning_scene.display(q=q)
            
            if (q < self.constraint.lb).any() or (q > self.constraint.ub).any():
                continue
            if self.constraint.planning_scene.is_valid(q) is False:
                continue
            # print('fwd:', constraint.forward_kinematics(q)[0])
            return q

class IKRegionSampler(RegionSampler):
    def __init__(self, constraint:ConstraintIKBase) -> None:
        super().__init__(constraint)
        self.pos_list = []
        self.quat_list = []
    
    def set_from_grasp_poses(self, cgc, c_idx, c, target_x):
        pos_list = []
        quat_list = []
        for i in range(3):
            pos, quat = cgc.get_global_grasp(c_idx[i], c[i], target_x[0:3], target_x[3:7])
            pos_list.append(pos)
            quat_list.append(quat)

        pos_list.reverse()
        quat_list.reverse()

        self.set_target_pose(target_x)
        self.set_grasp_poses(pos_list, quat_list)

    def set_grasp_poses(self, pos_list, quat_list):
        self.pos_list = pos_list
        self.quat_list = quat_list

    def sample(self):
        assert self.pose is not None, 'set pose first'

        self.constraint.update_target(self.pose)

        q_total = np.zeros(21)
        while True:
            for i in range(len(self.pos_list)):
                while True:
                    q = self.constraint.sample()
                    arm_name = self.constraint.arm_names[i]
                    arm_idx = self.constraint.arm_indices[arm_name]
                    r, q_out = self.constraint.solve_arm_ik(arm_name, q[7*arm_idx:7*(arm_idx+1)], self.pos_list[i], self.quat_list[i])
                    # print(arm_name, arm_idx, i, r, q_out)

                    if r:
                        q_total[7*arm_idx:7*(arm_idx+1)] = q_out
                        break
            
            if self.constraint.planning_scene.is_valid(q_total):
                # print(self.constraint.function_ik(q_total))
                return q_total
    

class LatentRegionSampler(RegionSampler):
    def __init__(self, constraint_model:LatentModel, constraint:ConstraintIKBase, validity_model:LatentValidityModel, model_sample_count=512, new_config_threshold=1.5) -> None:
        super().__init__(constraint)
        self.priority_queue = PriorityQueue()
        self.unique = count()
        self.constraint_model = constraint_model
        self.validity_model = validity_model
        self.model_sample_count = model_sample_count
        self.new_config_threshold = new_config_threshold
        self.debug = False

    def set_target_pose(self, pose):
        # self.priority_queue.queue.clear()
        return super().set_target_pose(pose)

    def sample(self, timeout=-1.0, q0=None):
        assert self.pose is not None, 'set pose first'

        self.constraint.update_target(self.pose)
        trial = 0
        start_time = time.time()

        # initial guess
        if q0 is not None:
            r = self.constraint.solve_ik(q0, self.pose)
            if r is True:
                if (q0 < self.constraint.lb).any() or (q0 > self.constraint.ub).any() == False:
                    if self.constraint.planning_scene.is_valid(q0) is True:
                        return q0
            
        while True:
            if timeout > 0:
                if time.time() - start_time > timeout:
                    return None

            if self.priority_queue.empty():
                t1 = time.time()
                nq, nz = self.constraint_model.sample_with_estimated_validity_with_q(self.model_sample_count, self.validity_model)
                t2 = time.time()
                for q, z in zip(nq, nz):
                    f = self.constraint.function_ik(q)
                    self.priority_queue.put((np.linalg.norm(f), next(self.unique), q, z))
                t3 = time.time()
                if self.debug:
                    print('sample', t2-t1, 'put', t3-t2)
            f,_, q,z = self.priority_queue.get()

            new_f = self.constraint.function_ik(q)
            
            if np.linalg.norm(new_f) > np.linalg.norm(f) + self.new_config_threshold:
                self.priority_queue.queue.clear()
                continue
            
            trial += 1
            # f = self.constraint.function_ik(q)
            # J = self.constraint.jacobian_ik(q)
            # J_inv = np.linalg.pinv(J)
            # xd = J_inv@f

            # print(J)
            # print(J_inv)
            # print(xd)
            # print('f', f)
            r = self.constraint.solve_ik(q, self.pose)
            # print('################ f=', self.constraint.function_ik(q), '######################')
            if r is False:
                continue

            
            if self.debug:
                print(f, q, r, trial)
                self.constraint.planning_scene.display(q=q)
            # self.constraint.planning_scene.display(q=q)
            # print(self.constraint.function(q))
            # print(self.constraint.function_ik(q))
            
            if (q < self.constraint.lb).any() or (q > self.constraint.ub).any():
                continue
            if self.constraint.planning_scene.is_valid(q) is False:
                if self.debug:
                    self.constraint.planning_scene.print_current_collision_infos()
                continue
            # print('fwd:', constraint.forward_kinematics(q)[0])
            return q


class MultipleRegionSampler(RegionSampler):
    def __init__(self):
        self.samplers = []
        self.sampler_weights = []
        self.p_sampler = []

    def add_sampler(self, sampler:RegionSampler, weight=1.0):
        self.samplers.append(sampler)
        self.sampler_weights.append(weight)
        self.p_sampler = np.array(self.sampler_weights)/np.sum(self.sampler_weights)

    def sample(self):
        idx = np.random.choice(len(self.samplers), p=self.p_sampler)
        return self.samplers[idx].sample()

class IntersectionRegionSampler(object):
    def __init__(self, constraint1:ConstraintIKBase, constraint2:ConstraintIKBase, **kwargs) -> None:
        self.constraint1 = constraint1
        self.constraint2 = constraint2

    def check_validity(self, q):
        if (q < self.constraint.lb).any() or (q > self.constraint.ub).any():
            return False
        if self.constraint.planning_scene.is_valid(q) is False:
            return False
        return True

class LatentIntersectionRegionSampler(IntersectionRegionSampler):
    def __init__(self, constraint_model1:LatentModel, constraint_model2:LatentModel, 
                 validity_model1:LatentValidityModel, validity_model2:LatentValidityModel,
                 constraint1:ConstraintIKBase, constraint2:ConstraintIKBase, **kwargs) -> None:
        super().__init__(constraint1, constraint2, **kwargs)

        self.constraint_model1 = constraint_model1
        self.constraint_model2 = constraint_model2

        self.validity_model1 = validity_model1
        self.validity_model2 = validity_model2

        self.priority_queue = PriorityQueue()
        self.unique = count()
        self.model_sample_count = 128
    
    def check_validity(self, q, constraint):
        if (q < constraint.lb).any() or (q > constraint.ub).any():
            return False
        if constraint.planning_scene.is_valid(q) is False:
            return False
        return True

    
    def sample(self):
        tick = False
        best_z1 = np.zeros(self.constraint_model1.z_dim)
        best_z2 = np.zeros(self.constraint_model2.z_dim)
        
        var = 1.0
        # first_trial = True
        while True:
            if self.priority_queue.empty():
                t1 = time.time()
                nq1, nz1 = self.constraint_model1.sample_from_z_with_estimated_validity_with_q(best_z1, var, self.model_sample_count, self.validity_model1)
                nq2, nz2 = self.constraint_model2.sample_from_z_with_estimated_validity_with_q(best_z2, var, self.model_sample_count, self.validity_model2)
                t3 = time.time()
                x1_set = []
                x2_set = []
                # import pdb; pdb.set_trace()
                for q, z in zip(nq1, nz1):
                    r = self.constraint1.project(q)
                    if r is False:
                        # print('??')
                        continue
                    if self.check_validity(q, self.constraint1) is True:
                        x = self.constraint1.get_object_pose(q)
                        x1_set.append((x,q,z))

                for q, z in zip(nq2, nz2):
                    r = self.constraint2.project(q)
                    if r is False:
                        # print('??')
                        continue
                    if self.check_validity(q, self.constraint2) is True:
                        x = self.constraint2.get_object_pose(q)
                        x2_set.append((x,q,z))

                # for q1, q2 in zip(nq1, nq2):
                #     self.constraint1.project(q1)
                #     if self.check_validity(q1, self.constraint1) is True:
                #         x1_set.append(self.constraint1.get_object_pose(q1))
                    
                #     self.constraint2.project(q2)
                #     if self.check_validity(q2, self.constraint2) is True:
                #         x2_set.append(self.constraint2.get_object_pose(q2))
                t4 = time.time()

                # print(nq1, nq2)
                for i in range(len(x1_set)):
                    for j in range(len(x2_set)):
                        
                        x1, q1, z1 = x1_set[i]
                        x2, q2, z2 = x2_set[j]
                        x3 = (x1 + x2) / 2.0 # midpoint btw two poses
                        
                        d_p = np.linalg.norm(x1[:3] - x2[:3])
                        if d_p > 0.5:
                            continue
                        # quaternion distance
                        quat1 = pyquaternion.Quaternion(x=x1[3], y=x1[4], z=x1[5], w=x1[6])
                        quat2 = pyquaternion.Quaternion(x=x2[3], y=x2[4], z=x2[5], w=x2[6])
                        quat3 = pyquaternion.Quaternion.slerp(quat1, quat2, 0.5)
                        d_q = pyquaternion.Quaternion.absolute_distance(quat1, quat2)

                        x3[3:6] = quat3.vector
                        x3[6] = quat3.scalar
                        
                        # print(d_p, d_q)

                        d = d_p + d_q * 1.0

                        if d_q > 0.5: 
                            continue

                        self.priority_queue.put((d, next(self.unique), x1, x2, x3, q1, q2, z1, z2))
                t5 = time.time()
                tick = True
                print('d', d, 'sample', t3-t1, 'project', t4-t3, 'put', t5-t4)
            else:
                d, _, x1, x2, x3, q1, q2, z1, z2 = self.priority_queue.get()
                # self.priority_queue.queue.clear()
                # constraint1 and q1 is fixed, pose: x1
                # print(d)
                # import pdb; pdb.set_trace()
                # print('q1', q1)
                # print('q2', q2)
                # if self.check_validity(q1,self.constraint1):
                # print('try 1')
                if tick:
                    tick = False
                    best_z1 = z1
                    best_z2 = z2
                    var = 0.3
                    print('x1', x1)
                    print('x2', x2)
                    print('x3', x3)
                q1_new = copy.deepcopy(q1) # not to break the original q1
                q2_new = copy.deepcopy(q2) # not to break the original q2

                r = self.constraint2.solve_ik(q2_new, x3)
                if r:
                    print('1')
                    r = self.constraint1.solve_ik(q1_new, x3)
                    if r:
                        print('2')
                        if self.check_validity(q1_new, self.constraint1) and self.check_validity(q2_new, self.constraint2):
                            return x3, q1_new, q2_new, z1, z2

                # self.constraint2.update_target(x1)
                # f = self.constraint2.function_ik(q2)
                # print('1-',f)
                # r = self.constraint2.project_ik(q2_new)
                # print('11-',r, q2_new)
                q2_new = copy.deepcopy(q2) # not to break the original q2
                t6 = time.time()
                r = self.constraint2.solve_ik(q2_new, x1)
                # print('q2_new', q2_new)
                # t7 = time.time()
                # print('solve 2', t7-t6)
                if r:
                    print('q2_new', q2_new)
                    if self.check_validity(q2_new, self.constraint2):
                        return x1, q1, q2_new, z1, z2
                    else:
                        print(q2_new<self.constraint1.lb)
                        print(q2_new>self.constraint1.ub)
                        self.constraint2.planning_scene.display(q2)
                        self.constraint2.planning_scene.print_current_collision_infos()
                    
                # constraint2 and q2 is fixed, pose: x2
                # if self.check_validity(q2,self.constraint2):
                # print('try 2')
                
                # self.constraint1.update_target(x2)
                # f = self.constraint1.function_ik(q1)
                # print('2-',  f)
                # r = self.constraint1.project_ik(q1)
                # print('12-',r, q1)
                t8 = time.time()
                r = self.constraint1.solve_ik(q1, x2)
                t9 = time.time()
                # print('solve 1', t9-t8)
                if r:
                    print('q1_new', q1)
                    if self.check_validity(q1, self.constraint1):
                        return x2, q1, q2, z1, z2
                    else:
                        print(q1<self.constraint1.lb)
                        print(q1>self.constraint1.ub)
                        print(self.constraint1.function(q1))
                        self.constraint1.planning_scene.display(q1)
                        self.constraint1.planning_scene.print_current_collision_infos()

                # import pdb; pdb.set_trace()
