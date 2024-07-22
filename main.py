from inp_parser import inp_parser
from solver import OptimFEM
import numpy as np
import torch
from util import demo_results


def form_hole_1():
    inp_file = 'inp/hole-pyfem.inp'
    my_parser = inp_parser()
    all_nodes = my_parser.parser_nodes(inp_file)
    all_elements = my_parser.parser_elements(inp_file)
    node_set_1 = my_parser.parser_node_set(inp_file, 'Set-1', 'instance')
    node_set_2 = my_parser.parser_node_set(inp_file, 'Set-2', 'instance')
    problem = OptimFEM(all_nodes, all_elements,
                       material={'type': 'plane_stress', 'E': 1e9, 'nu' :0.3},
                       bcs=[[node_set_1, 'u', np.zeros(shape=len(node_set_1))], [node_set_1, 'v', np.zeros(shape=len(node_set_1))],
                            [node_set_2, 'u', 5.0 * np.ones(shape=len(node_set_2))],  [node_set_2, 'v', np.zeros(shape=len(node_set_2))]])
    optimizer = torch.optim.AdamW(problem.parameters(), lr=1e-1, weight_decay=1e-4)
    num_epoch = 100000
    problem.train()
    for n in range(num_epoch):
        # 前向传播
        loss = problem()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n % 20 == 0:
            print(loss.item())
            demo_results(np.array(all_nodes), np.array([[problem.disp_vect[2*i].cpu().detach().numpy(),
                                                problem.disp_vect[2*i+1].cpu().detach().numpy()] for i in range(problem.num_nodes)])[:, :, 0],
                         save='results//'+str(n)+'.png')

if __name__ == '__main__':
    form_hole_1()
