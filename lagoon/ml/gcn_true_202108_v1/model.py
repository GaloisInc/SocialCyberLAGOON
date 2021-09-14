
import lagoon.db.schema as sch

import math
import pyobjconfig as pc
import pyobjconfig.torch
import torch

def Act():
    #return torch.nn.PReLU()
    #return torch.nn.GELU()
    return torch.nn.ReLU()


def Linear(nin, nout, *, zero_wt=False, **kwargs):
    l = torch.nn.Linear(nin, nout, **kwargs)
    if zero_wt:
        l.weight.data.zero_()
    else:
        torch.nn.init.kaiming_uniform_(l.weight.data)
    l.bias.data.zero_()
    return l


def Norm(nf, **kwargs):
    #return torch.nn.Sequential()
    #l = torch.nn.BatchNorm1d(nf, **kwargs)

    # See https://arxiv.org/pdf/1911.07013.pdf
    elementwise = False
    l = torch.nn.LayerNorm(nf, elementwise_affine=elementwise, **kwargs)
    if elementwise:
        l.weight.data.fill_(1)
        l.bias.data.zero_()
    return l



class Model(pc.torch.ConfigurableModule):
    class config(pc.PydanticBaseModel):
        batch_size: int = 16
        # Avoid bool: https://github.com/guildai/guildai/issues/124
        data_badwords: int = 1
        data_cheat: int = 0
        data_type: int = 1
        embed_size: int = 32
        hops: int = 1
        lr: float = 1e-4
        train_epochs: int = 10000
        type_embed_size: int = 4
        window_size: float = 0.5 * 3600 * 24 * 365.25


    def build(self):
        self.type_embeds = torch.nn.Embedding(len(sch.EntityTypeEnum),
                self.config.type_embed_size)
        self.preproc = torch.nn.Sequential(
                Linear(self.config.embed_size, self.config.embed_size))
        hop_layers = []
        for _ in range(4):
            hop_layers.append(ModelGcnHop(self.config.embed_size))
            hop_layers.append(ModelGcnMlp(self.config.embed_size))
        self.hops = torch.nn.Sequential(*hop_layers)
        self.model_out = torch.nn.Sequential(
                Norm(self.config.embed_size),
                Act(),
                Linear(self.config.embed_size, 1, zero_wt=True),
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=self.config.lr,
                betas=(0.9, 0.95))


    def forward(self, batch):
        """Given some input [(adj_matrix, ent_props), ...], make a prediction.

        Note that `ent_props` comes from model! Specifically, `type_embeds`
        """
        adj = [b[0] for b in batch]
        x = torch.cat([b[1] for b in batch], 0)
        # b[2] is stats

        x = self.preproc(x)
        x, adj = self.hops((x, adj))
        # For BN, trim after computation instead of before
        x = self.model_out(x)
        r = []
        i = 0
        for a in adj:
            j = i + a.size(0)
            r.append(x[i])
            i = j
        assert i == x.size(0), f'{i} == {x.size(0)}'

        r = torch.stack(r)
        assert len(adj) == r.size(0), r.size()
        assert 1 == r.size(1), r.size()
        return r



class ModelGcnHop(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.setup = torch.nn.Sequential(
                Norm(embed_size),
                Act(),
        )
        layers = [
                Norm(embed_size),
                Act(),
                Linear(embed_size, embed_size, zero_wt=True),
                torch.nn.Dropout(0.1),
        ]

        self.multihead = True
        if not self.multihead:
            layers.insert(0, Linear(embed_size, embed_size))
        elif self.multihead:
            # TODO parallel heads have same benefit as in resnet
            self.mh_q = Linear(embed_size, embed_size)
            self.mh_k = Linear(embed_size, embed_size)
            self.mh_v = Linear(embed_size, embed_size)

        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x_parts):
        x, x_adj = x_parts

        x_setup = self.setup(x)

        i = 0
        xx = []
        for a in x_adj:
            j = i + a.size(0)
            ax = x_setup[i:j]
            if not self.multihead:
                # Traditional GCN
                asum = torch.sparse.sum(a, 1).to_dense()[:, None]
                xx.append((a @ ax) / asum)
            else:
                # SM-based transformer-style GCN
                q = self.mh_q(ax)
                k = self.mh_k(ax)
                v = self.mh_v(ax)

                if False:
                    # Bad, no adjacency
                    dot = (q @ k.transpose(-1, -2))
                    dot /= k.size(-1) ** 0.5
                    dot = dot.softmax(-1)
                    xx.append(dot @ v)
                else:
                    # Good, adjacency
                    # Want: a * (q @ k.T) = ?? Seems hard to represent.
                    # Worse, pytorch doesn't support sparse, sparse -> sparse
                    # matmul
                    if True:
                        # Non-sparse ; more efficient as long as communication
                        # overhead is sufficiently large, which seems to often
                        # be the case.
                        dot = q @ k.transpose(-1, -2)
                        dot_m = torch.empty_like(dot).fill_(-math.inf)
                        assert a.shape == dot.shape
                        a = a.to_dense()
                        a_yes = (a != 0)
                        dot /= a.sum(1, keepdim=True).pow_(0.5)
                        dot_m[a_yes] = dot[a_yes]
                        dot_m = dot_m.softmax(-1)
                        xx.append(dot_m @ v)
                    else:
                        # Sparse; less efficient on e.g. GPUs in practice.
                        arows = [ar.coalesce().indices()[0] for ar in a]
                        kt = k.transpose(-1, -2)
                        # Complex because each row may be a different size
                        rows = []
                        for qi, arow in enumerate(arows):
                            dot = q[qi, None] @ kt[:, arow]
                            dot /= arow.size(0) ** 0.5
                            assert dot.shape == (1, arow.size(0)), dot.shape
                            dot = dot.softmax(-1)
                            rows.append(dot @ v[arow])
                        xx.append(torch.cat(rows, 0))

                if False:
                    # TODO Append number of relations since sm-based (or trad)
                    # can't represent that
                    pass
            i = j
        assert i == x.size(0), f'{i} == {x.size(0)}'
        xx = torch.cat(xx, 0)
        return x + self.layers(xx), x_adj



class ModelGcnMlp(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.layers = torch.nn.Sequential(*[
                Norm(embed_size),
                Act(),
                Linear(embed_size, embed_size*4),
                Norm(embed_size*4),
                Act(),
                Linear(embed_size*4, embed_size, zero_wt=True),
                torch.nn.Dropout(0.1),
        ])

    def forward(self, x):
        x, x_adj = x
        return x + self.layers(x), x_adj

