import torch
from torch.autograd import Function

class one_nn(Function):
    @staticmethod
    def forward(ctx, ref, query):
        """
        one nn calculation

        Args:
            ctx ([type]): [description]
            ref (Torch.Tensor): target with the size of (n, 3)
            query (Torch.Tensor): prediction with the size of (n, 3)

        Returns:
            [type]: index of target
        """
        ind = []
        # print(ref.shape, query.shape)
        # exit()

        num_p = query.size(0)
        size = 5000

        if int(num_p / size) > 0 and num_p % size > 0:
            num_loop = int(num_p / size) + 1
        elif int(num_p / size) == 0:
            num_loop = 1
        else:
            num_loop = int(num_p / size)
        
        if int(num_p / size) == 0:
            size = num_p

        ref = ref.detach()  # target
        query = query.detach()  # pred

        ref = ref.unsqueeze(0).repeat(size, 1, 1)

        for i in range(num_loop):
            if i == num_loop - 1 and num_p % size > 0:
                query_t = query[i * size:i * size + num_p % size].unsqueeze(1).repeat(1, ref.size(1), 1)
                ref = ref[0].unsqueeze(0).repeat(num_p % size, 1, 1)
            else:
                query_t = query[i * size:(i + 1) * size].unsqueeze(1).repeat(1, ref.size(1), 1)
            
            dist = torch.norm(ref - query_t, dim=2)
            ind_t = dist.topk(1, largest=False).indices.view(-1)
            ind.append(ind_t)

        ind = torch.cat(ind)
        return ind
