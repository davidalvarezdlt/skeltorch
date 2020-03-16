import torch.nn
import numpy as np
import scipy.linalg


class GlowModelCouplingNetwork(torch.nn.Module):
    n_in = None
    n_out = None
    num_filters = None
    kernel_size = None
    scale_out = None

    def __init__(self, n_in, n_out, num_filters, kernel_size, scale_out=True):
        super(GlowModelCouplingNetwork, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.scale_out = scale_out
        self.coupling_network_layers = torch.nn.Sequential(
            torch.nn.Conv2d(n_in, num_filters, kernel_size, padding=kernel_size // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_filters, num_filters, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_filters, n_out, kernel_size, padding=kernel_size // 2),
        )
        self.scale = torch.nn.Parameter(torch.zeros(1, n_out, 1, 1))

    def forward(self, h):
        h = self.coupling_network_layers.forward(h)
        return h * torch.exp(self.scale * 3) if self.scale_out else h


class GlowModelCoupling(torch.nn.Module):
    coupling_type = None
    num_filters = None
    kernel_size = None

    def __init__(self, num_channels, coupling_type, num_filters, kernel_size):
        super(GlowModelCoupling, self).__init__()
        self.num_channels = num_channels
        self.coupling_type = coupling_type
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        if self.coupling_type == 'affine':
            self.coupling_network = GlowModelCouplingNetwork(
                self.num_channels // 2, self.num_channels, self.num_filters, self.kernel_size
            )
        elif self.coupling_type == 'additive':
            self.coupling_network = GlowModelCouplingNetwork(
                self.num_channels // 2, self.num_channels // 2, self.num_filters, self.kernel_size
            )

    def forward(self, h):
        h1, h2 = torch.chunk(h, 2, dim=1)
        aux = self.coupling_network.forward(h1)
        if self.coupling_type == 'affine':
            s, t = torch.chunk(aux, 2, dim=1)
            s = torch.sigmoid(s + 2)
            return torch.cat([h1, s * (h2 + t)], dim=1), s.log().view(h.size(0), -1).sum(1)
        elif self.coupling_type == 'additive':
            return torch.cat([h1, h2 + aux], dim=1), torch.zeros(h.size(0)).to(h.device)

    def reverse(self, h):
        h1, h2 = torch.chunk(h, 2, dim=1)
        aux = self.coupling_network.forward(h1)
        if self.coupling_type == 'affine':
            s, t = torch.chunk(aux, 2, dim=1)
            s = torch.sigmoid(s + 2)
            return torch.cat([h1, h2 / s - t], dim=1)
        elif self.coupling_type == 'additive':
            return torch.cat([h1, h2 - aux], dim=1)


class GlowModelMixer(torch.nn.Module):
    num_channels = None
    permutation_type = None

    def __init__(self, num_channels, permutation_type):
        super(GlowModelMixer, self).__init__()
        self.num_channels = num_channels
        self.permutation_type = permutation_type
        self._init_mixer()

    def _init_mixer(self):
        if self.permutation_type == 'conv':
            self._init_mixer_conv()
        elif self.permutation_type == 'shuffle':
            self._init_mixer_shuffle()
        elif self.permutation_type == 'reversed':
            self._init_mixer_reversed()

    def _init_mixer_conv(self):
        weight = np.random.randn(self.num_channels, self.num_channels)
        q, _ = scipy.linalg.qr(weight)
        w_p, w_l, w_u = scipy.linalg.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        self.register_buffer('w_p', torch.from_numpy(w_p))
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.register_buffer('s_sign', torch.sign(torch.from_numpy(w_s)))
        self.w_l = torch.nn.Parameter(torch.from_numpy(w_l))
        self.log_w_s = torch.nn.Parameter(torch.log(1e-7 + torch.abs(torch.from_numpy(w_s))))
        self.w_u = torch.nn.Parameter(torch.from_numpy(w_u))

    def _init_mixer_shuffle(self):
        self.register_buffer('indices', torch.randperm(self.num_channels))
        self.register_buffer('indices_reversed', torch.argsort(self.indices))

    def _init_mixer_reversed(self):
        self.register_buffer('indices', torch.arange(self.num_channels - 1, -1, -1))
        self.register_buffer('indices_reversed', torch.arange(self.num_channels - 1, -1, -1))

    def calc_conv_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ (self.w_u * self.u_mask + torch.diag(self.s_sign * torch.exp(self.log_w_s)))
        )
        return weight

    def forward(self, h):
        if self.permutation_type == 'conv':
            weight = self.calc_conv_weight().unsqueeze(2).unsqueeze(3)
            return torch.nn.functional.conv2d(h, weight), self.log_w_s.sum() * h.size(2) * h.size(3)
        else:
            return h[:, self.indices], 0

    def reverse(self, h):
        if self.permutation_type == 'conv':
            invweight = self.calc_conv_weight().inverse().unsqueeze(2).unsqueeze(3)
            return torch.nn.functional.conv2d(h, invweight)
        else:
            return h[:, self.indices_reversed]


class GlowModelActNorm(torch.nn.Module):
    num_channels = None

    def __init__(self, num_channels):
        super(GlowModelActNorm, self).__init__()
        self.num_channels = num_channels
        self.t = torch.nn.Parameter(torch.zeros(1, self.num_channels, 1, 1))
        self.logs = torch.nn.Parameter(torch.zeros(1, self.num_channels, 1, 1))
        self.register_buffer('initialized', torch.tensor(0).bool())

    def _init_act_norm(self, h):
        flatten = h.permute(1, 0, 2, 3).contiguous().view(h.size(1), -1).data
        self.t.data = -flatten.mean(1).view(1, -1, 1, 1)
        self.logs.data = torch.log(1 / (flatten.std(1) + 1e-7)).view(1, -1, 1, 1)
        self.initialized = torch.tensor(1).bool()

    def forward(self, h):
        if not self.initialized.all():
            self._init_act_norm(h)
        return torch.exp(self.logs) * (h + self.t), self.logs.sum() * h.size(2) * h.size(3)

    def reverse(self, h):
        return h * torch.exp(-self.logs) - self.t


class GlowModelFlow(torch.nn.Module):

    def __init__(self, num_channels, permutation_type, coupling_type, num_filters, kernel_size):
        super(GlowModelFlow, self).__init__()
        self.flow_act_norm = GlowModelActNorm(num_channels)
        self.flow_mixer = GlowModelMixer(num_channels, permutation_type)
        self.flow_coupling = GlowModelCoupling(num_channels, coupling_type, num_filters, kernel_size)

    def forward(self, h):
        h, log_det_act_norm = self.flow_act_norm.forward(h)
        h, log_det_mixer = self.flow_mixer(h)
        h, log_det_coupling = self.flow_coupling(h)
        return h, log_det_act_norm + log_det_mixer + log_det_coupling

    def reverse(self, h):
        h = self.flow_coupling.reverse(h)
        h = self.flow_mixer.reverse(h)
        return self.flow_act_norm.reverse(h)


class GlowModelSqueezer(torch.nn.Module):
    squeezing_factor = None

    def __init__(self, squeezing_factor):
        super(GlowModelSqueezer, self).__init__()
        self.squeezing_factor = squeezing_factor

    def forward(self, h):
        batch_size, num_channels, height, width = h.size()
        h = h.view(batch_size, num_channels, height // self.squeezing_factor, self.squeezing_factor,
                   width // self.squeezing_factor, self.squeezing_factor)
        h = h.permute(0, 1, 3, 5, 2, 4).contiguous()
        return h.view(batch_size, num_channels * (self.squeezing_factor ** 2), height // self.squeezing_factor,
                      width // self.squeezing_factor)

    def reverse(self, h):
        batch_size, num_channels, height, width = h.size()
        h = h.view(batch_size, num_channels // (self.squeezing_factor ** 2), self.squeezing_factor,
                   self.squeezing_factor, height, width)
        h = h.permute(0, 1, 4, 2, 5, 3).contiguous()
        return h.view(batch_size, num_channels // (self.squeezing_factor ** 2), height * self.squeezing_factor,
                      width * self.squeezing_factor)


class GlowModelBlock(torch.nn.Module):

    def __init__(self, num_channels, num_flows, squeezing_factor, permutation_type, coupling_type, num_filters,
                 kernel_size):
        super(GlowModelBlock, self).__init__()
        self.block_squeezer = GlowModelSqueezer(squeezing_factor)
        self.block_flows = torch.nn.ModuleList()
        num_channels *= squeezing_factor ** 2
        for _ in range(num_flows):
            self.block_flows.append(
                GlowModelFlow(num_channels, permutation_type, coupling_type, num_filters, kernel_size)
            )

    def forward(self, h):
        h = self.block_squeezer.forward(h)
        log_det = torch.zeros(h.size(0)).to(h.device)
        for block_flow in self.block_flows:
            h, log_det_flow = block_flow.forward(h)
            log_det += log_det_flow
        h, z = torch.chunk(h, 2, dim=1)
        return h, z, log_det

    def reverse(self, h, z):
        h = torch.cat([h, z], dim=1)
        for block_flow in self.block_flows[::-1]:
            h = block_flow.reverse(h)
        return self.block_squeezer.reverse(h)


class GlowModel(torch.nn.Module):
    num_channels = None

    def __init__(self, num_channels, num_blocks, num_flows, squeezing_factor, permutation_type, coupling_type,
                 num_filters, kernel_size):
        super(GlowModel, self).__init__()
        self.num_channels = num_channels
        self.squeezing_factor = squeezing_factor
        self.glow_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.glow_blocks.append(
                GlowModelBlock(
                    num_channels, num_flows, squeezing_factor, permutation_type, coupling_type, num_filters, kernel_size
                )
            )
            num_channels *= squeezing_factor ** 2
            num_channels = num_channels // 2

    def forward(self, x):
        h = x
        z_list = []
        log_det = torch.zeros(h.size(0)).to(h.device)
        for glow_block in self.glow_blocks:
            h, z, log_det_block = glow_block.forward(h)
            z_list.append(z.view(z.size(0), -1))
            log_det += log_det_block
        z_list.append(h.view(z.size(0), -1))
        return torch.cat(z_list, dim=1), log_det

    def reverse(self, z):
        z_list = self._get_blockwise(z)
        h = z_list[-1]
        for glow_block, z in zip(self.glow_blocks[::-1], z_list[-2::-1]):
            h = glow_block.reverse(h, z)
        return h

    def _get_blockwise(self, z):
        z_list = []
        block_channels = self.num_channels
        block_size = 0
        block_im_size = np.sqrt(z.size(1) / self.num_channels).astype(np.int)
        block_index_start = 0
        for block_n in range(len(self.glow_blocks)):
            block_channels = block_channels * self.squeezing_factor ** 2
            block_channels = block_channels // 2
            block_im_size = block_im_size // 2
            block_size = block_channels * block_im_size ** 2
            z_list.append(z[:, block_index_start:block_index_start + block_size]
                          .view(z.size(0), block_channels, block_im_size, block_im_size))
            block_index_start += block_size
        z_list.append(z[:, block_index_start:block_index_start + block_size]
                      .view(z.size(0), block_channels, block_im_size, block_im_size))
        return z_list
