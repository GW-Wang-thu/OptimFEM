import torch
import numpy as np
import torch.nn as nn


def assemble_element(node_list, element_list, dim=2):
    coord_list = []
    for tmp_nodes in element_list:
        tmp_coord = []
        for node_id in tmp_nodes:
            tmp_coord.append(node_list[node_id-1])
        if type(node_list) == type([]):
            coord_list.append(tmp_coord)
        else:
            coord_list.append(torch.cat(tmp_coord))
    if type(node_list) == type([]):
        return np.array(coord_list).swapaxes(1, 2) # (N, 2, 4)
    else:
        return torch.stack(coord_list, dim=0)  # (N, 2, 4)


class OptimFEM(nn.Module):
    def __init__(self, all_nodes, all_elements, material, bcs, dim=2, device='cuda'):
        super(OptimFEM, self).__init__()
        self.all_nodes = all_nodes
        self.all_elements = all_elements
        self.num_nodes = len(self.all_nodes)
        self.num_elements = len(self.all_elements)
        self.material_param = material
        self.material = torch.from_numpy(self.__init_material__(material).astype('float32')).to(device).unsqueeze(0).repeat_interleave(self.num_elements, dim=0)
        self.device = device
        self.dim = dim
        # self.disp_vect = nn.Parameter(torch.randn(self.num_nodes,  self.dim, requires_grad=True, dtype=torch.float32).to(device))
        self.disp_vect = nn.ParameterList([nn.Parameter(torch.randn(1) * 0.1) for _ in range(self.num_nodes * 2)])
        # self.disp_array_element = assemble_element(self.disp_vect, self.all_elements)
        # self.disp_bc, self.disp_bc_mask = self.__init_bcs__(bcs)
        self.bcs = bcs
        self.weight = 1  # weight of boundary loss to total strain energy
        self.gauss_points_xi = [-1/np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]
        self.gauss_points_eta = [-1/np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]
        self.gauss_weights = [1, 1, 1, 1]
        self.num_gauss_point = 4
        self.element_coord_list = assemble_element(self.all_nodes, self.all_elements)
        # self.jacobi, self.jacobi_inv = self.__init_calc_jacobi__()
        # self.B = torch.from_numpy(self.__init_calc_strainform__().astype('float32')).to(device)
        self.debug = False

    def __init_material__(self, material):
        if material['type'] == 'plane_stress':
            E = material['E']
            nu = material['nu']
            D = (E/(1-nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        else:
            D=None
        return D
    #
    # def __init_bcs__(self, bcs):
    #     disp_vect = torch.zeros_like(self.disp_vect, requires_grad=False)
    #     disp_mask_vect = torch.zeros_like(self.disp_vect, requires_grad=False)
    #     for bc_set in bcs:
    #         tmp_node_id, tmp_opt, tmp_bc = bc_set
    #         if tmp_opt == 'u':
    #             for i in range(len(tmp_node_id)):
    #                 disp_vect[tmp_node_id[i]-1, 0] = tmp_bc[i]
    #                 disp_mask_vect[tmp_node_id[i]-1, 0] = 1.0
    #         elif tmp_opt == 'v':
    #             for i in range(len(tmp_node_id)):
    #                 disp_vect[tmp_node_id[i] - 1, 1] = tmp_bc[i]
    #                 disp_mask_vect[tmp_node_id[i] - 1, 1] = 1.0
    #     return disp_vect, disp_mask_vect

    def __calc_strainform__(self, x1, x2, x3, x4, y1, y2, y3, y4):
        B = [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]]
        # 计算雅可比矩阵
        for i in range(4):  # 积分点遍历相加
            tmp_dNdxi, tmp_dNdeta = self.__shape_function_derivatives__(xi=self.gauss_points_xi[i], eta=self.gauss_points_eta[i])   # (4, ), (4, )
            a = (tmp_dNdxi[0] * x1 + tmp_dNdxi[1] * x2 + tmp_dNdxi[2] * x3 + tmp_dNdxi[3] * x4)
            b = (tmp_dNdxi[0] * y1 + tmp_dNdxi[1] * y2 + tmp_dNdxi[2] * y3 + tmp_dNdxi[3] * y4)
            c = (tmp_dNdeta[0] * x1 + tmp_dNdeta[1] * x2 + tmp_dNdeta[2] * x3 + tmp_dNdeta[3] * x4)
            d = (tmp_dNdeta[0] * y1 + tmp_dNdeta[1] * y2 + tmp_dNdeta[2] * y3 + tmp_dNdeta[3] * y4)

            det_j = a * d - b * c
            inv_m = 1.0 / det_j
            inv_jacobi_array_11 = inv_m * d
            inv_jacobi_array_12 = - inv_m * c
            inv_jacobi_array_21 = - inv_m * b
            inv_jacobi_array_22 = inv_m * a
            for j in range(4):
                B[0][2 * j] += self.gauss_weights[i] * (inv_jacobi_array_11 * tmp_dNdxi[j] + inv_jacobi_array_12 * tmp_dNdeta[j])
                B[1][2 * j + 1] += self.gauss_weights[i] * (inv_jacobi_array_21 * tmp_dNdxi[j] + inv_jacobi_array_22 * tmp_dNdeta[j])
                B[2][2 * j] += self.gauss_weights[i] * (inv_jacobi_array_11 * tmp_dNdxi[j] + inv_jacobi_array_12 * tmp_dNdeta[j])
                B[2][2 * j + 1] += self.gauss_weights[i] * (inv_jacobi_array_21 * tmp_dNdxi[j] + inv_jacobi_array_22 * tmp_dNdeta[j])
            # assert(B[0][0].grad is not None)
        return B    # (3, 8)

    def __calc_element_energy__(self):
        energy = 0.0
        E, nu = self.material_param['E'], self.material_param['nu']
        for i in range(self.num_elements):  # 逐个单元计算应变能然后叠加
            tmp_element_into = self.all_elements[i]
            epsilon_xx, epsilon_yy, gamma_xy = 0.0, 0.0, 0.0
            tmp_B = self.__calc_strainform__(x1=self.disp_vect[(tmp_element_into[0]-1) * 2] + self.all_nodes[tmp_element_into[0]-1][0],
                                             x2=self.disp_vect[(tmp_element_into[0]-1) * 2] + self.all_nodes[tmp_element_into[1]-1][0],
                                             x3=self.disp_vect[(tmp_element_into[2]-1) * 2] + self.all_nodes[tmp_element_into[2]-1][0],
                                             x4=self.disp_vect[(tmp_element_into[3]-1) * 2] + self.all_nodes[tmp_element_into[3]-1][0],
                                             y1=self.disp_vect[(tmp_element_into[0]-1) * 2 + 1] + self.all_nodes[tmp_element_into[0]-1][1],
                                             y2=self.disp_vect[(tmp_element_into[1]-1) * 2 + 1] + self.all_nodes[tmp_element_into[1]-1][1],
                                             y3=self.disp_vect[(tmp_element_into[2]-1) * 2 + 1] + self.all_nodes[tmp_element_into[2]-1][1],
                                             y4=self.disp_vect[(tmp_element_into[3]-1) * 2 + 1] + self.all_nodes[tmp_element_into[3]-1][1])
            for j in range(4):
                epsilon_xx = tmp_B[0][2 * j] * self.disp_vect[(tmp_element_into[j]-1) * 2 ] \
                             + tmp_B[0][2 * j + 1] * self.disp_vect[(tmp_element_into[j]-1) * 2 + 1]
                epsilon_yy = tmp_B[1][2 * j] * self.disp_vect[(tmp_element_into[j]-1) * 2 ] \
                             + tmp_B[1][2 * j + 1] * self.disp_vect[(tmp_element_into[j]-1) * 2 + 1]
                gamma_xy = tmp_B[2][2 * j] * self.disp_vect[(tmp_element_into[j]-1) * 2 ] \
                             + tmp_B[2][2 * j + 1] * self.disp_vect[(tmp_element_into[j]-1) * 2 + 1]
            sigma_xx = E / (1 - nu ** 2) * (epsilon_xx + nu * epsilon_yy)
            sigma_yy = E / (1 - nu ** 2) * (epsilon_yy + nu * epsilon_xx)
            sigma_xy = E / (1 + nu) * gamma_xy
            tmp_energy = 0.5 * (epsilon_xx * sigma_xx + epsilon_yy * sigma_yy + 2 * gamma_xy * sigma_xy)
            if self.debug:
                print(i, tmp_energy.item())
            energy += tmp_energy**2
        return energy

    def __shape_function__(self, xi, eta):
        N1 = (1 - xi) * (1 - eta) / 4
        N2 = (1 + xi) * (1 - eta) / 4
        N3 = (1 + xi) * (1 + eta) / 4
        N4 = (1 - xi) * (1 + eta) / 4
        return [N1, N2, N3, N4],

    def __shape_function_derivatives__(self, xi, eta):
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return dN_dxi, dN_deta

    def __calc_bc_loss__(self):
        loss = 0.0
        for bc_set in self.bcs:
            tmp_node_id, tmp_opt, tmp_bc = bc_set
            if tmp_opt == 'u':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2] - tmp_bc[i]) ** 2
            elif tmp_opt == 'v':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2+1] - tmp_bc[i]) ** 2
        return loss


    def forward(self, x=None):
        total_energy = self.__calc_element_energy__()
        bc_constrain = self.__calc_bc_loss__()
        return total_energy + bc_constrain * 1.0e10#(self.weight * total_energy.item())