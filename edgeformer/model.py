import copy
import math
import torch
from torch import nn
from torch.nn import Parameter, ModuleList, LayerNorm, Dropout
import torch.nn.functional as F
from torch.optim import AdamW, Adam

from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from torch_scatter import scatter_add



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# from the original Sinha et al. CLUTRR code
def get_mlp(input_dim, output_dim, num_layers=2, dropout=0.0):
    network_list = []
    assert num_layers > 0
    if num_layers > 1:
        for _ in range(num_layers - 1):
            network_list.append(nn.Linear(input_dim, input_dim))
            network_list.append(nn.ReLU())
            network_list.append(nn.Dropout(dropout))
    network_list.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(
        *network_list
    )


class EdgeAttentionFlat(nn.Module):
    def __init__(self, d_model, num_heads, dropout, lesion_scores=False, lesion_values=False):
        super(EdgeAttentionFlat, self).__init__()
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model, bias=False), 4)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes = query.size(1)

        k, v, q = [l(x) for l, x in zip(self.linears, (key, value, query))]
        k = k.view(num_batches, num_nodes, num_nodes, self.num_heads, self.d_k)
        v = v.view_as(k)
        q = q.view_as(k)

        scores_r = torch.einsum("bxyhd,bxzhd->bxyzh", q, k) / math.sqrt(self.d_k)
        scores_r = scores_r.masked_fill(mask.unsqueeze(4), -1e9)
        scores_l = torch.einsum("bxyhd,bzyhd->bxyzh", q, k) / math.sqrt(self.d_k)
        scores_l = scores_l.masked_fill(mask.unsqueeze(4), -1e9)
        scores = torch.cat((scores_r, scores_l), dim=3)

        att = F.softmax(scores, dim=3)
        att = self.dropout(att)
        att_r, att_l = torch.split(att, scores_r.size(3), dim=3)

        x_r = torch.einsum("bxyzh,bxzhd->bxyhd", att_r, v)
        x_l = torch.einsum("bxyzh,bzyhd->bxyhd", att_l, v)

        x = x_r + x_l
        x = torch.reshape(x, (num_batches, num_nodes, num_nodes, self.d_model))

        return self.linears[-1](x)


class EdgeAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, lesion_scores=False, lesion_values=False):
        super(EdgeAttention, self).__init__()
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model, bias=False), 6)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.lesion_scores = lesion_scores
        self.lesion_values = lesion_values

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes = query.size(1)

        left_k, right_k, left_v, right_v, query = [l(x) for l, x in zip(self.linears, (key, key, value, value, key))]
        left_k = left_k.view(num_batches, num_nodes, num_nodes, self.num_heads, self.d_k)
        right_k = right_k.view_as(left_k)
        left_v = left_v.view_as(left_k)
        right_v = right_v.view_as(left_k)
        query = query.view_as(left_k)

        if self.lesion_scores:
            query = right_k
            scores = torch.einsum("bxahd,bxyhd->bxayh", left_k, query) / math.sqrt(self.d_k)
        else:
            scores = torch.einsum("bxahd,bayhd->bxayh", left_k, right_k) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask.unsqueeze(4), -1e9)

        val = torch.einsum("bxahd,bayhd->bxayhd", left_v, right_v)

        att = F.softmax(scores, dim=2)
        att = self.dropout(att)
        if self.lesion_values:
            x = torch.einsum("bxayh,bxahd->bxyhd", att, left_v)
            x = x.contiguous()
            x = x.view(num_batches, num_nodes, num_nodes, self.d_model)
        else:
            x = torch.einsum("bxayh,bxayhd->bxyhd", att, val)
            x = x.view(num_batches, num_nodes, num_nodes, self.d_model)

        return self.linears[-1](x)


class EdgeTransformerLayer(nn.Module):

    def __init__(self, num_heads=4, dropout=0.2, dim=200, ff_factor=4, flat_attention=False, activation="relu"):
        super().__init__()

        self.num_heads = num_heads

        dropout = dropout

        d_model = dim
        d_ff = ff_factor * d_model

        self.flat_attention = flat_attention

        if self.flat_attention:
            self.edge_attention = EdgeAttentionFlat(d_model, self.num_heads, dropout, lesion_scores=False, lesion_values=False)
        else:
            self.edge_attention = EdgeAttention(d_model, self.num_heads, dropout, lesion_scores=False, lesion_values=False)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, batched_graphs, mask=None):

        batched_graphs = self.norm1(batched_graphs)
        batched_graphs2 = self.edge_attention(batched_graphs, batched_graphs, batched_graphs, mask=mask)
        batched_graphs = batched_graphs + self.dropout1(batched_graphs2)
        batched_graphs = self.norm2(batched_graphs)
        batched_graphs2 = self.linear2(self.dropout2(self.activation(self.linear1(batched_graphs))))
        batched_graphs = batched_graphs + self.dropout3(batched_graphs2)

        return batched_graphs


class EdgeTransformerEncoder(nn.Module):

    def __init__(self, h_index, t_index, r_index, num_heads, num_relation, num_nodes, dropout, dim, ff_factor, flat_attention, activation="relu"):
        super().__init__()

        self.num_heads = num_heads
        self.num_relation = num_relation
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.dim = dim
        self.ff_factor =ff_factor
        self.deep_residual = False
        self.share_layers = self.share_layers
        self.num_layers = self.num_message_rounds

        # last two embeddings are for queried_mask and unqueried_mask
        self.relation_emb = torch.nn.Embedding(num_embeddings=2 * self.num_relation + 2, embedding_dim=self.dim)

        # any pair of (i, j) can be indexed to relation embedding through pair_index
        pair_index = torch.tensor([2 * self.num_relation + 1] * (self.num_nodes * self.num_nodes), dtype=torch.int)
        indexing = h_index * self.num_nodes + t_index
        pair_index[indexing] = r_index
        self.pair_index = pair_index

        encoder_layer = EdgeTransformerLayer(num_heads, dropout, dim, ff_factor, flat_attention, activation=activation)
        self.layers = _get_clones(encoder_layer, self.num_layers)

        self._reset_parameters()

    def easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        remove_edge_mask = functional.as_mask(edge_index, graph.num_edge)
        return remove_edge_mask

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            remove_edge_mask = self.easy_edges(graph, h_index, t_index, r_index)
            remove_edge_index = graph.edge_list[:, 0] * self.num_nodes + graph.edge_list[:, 1]
            batch_pair_index = self.pair_index.clone()
            batch_pair_index[remove_edge_index[remove_edge_mask]] = self.num_relation + 1

        shape = h_index.shape
        assert graph.num_relation
        graph = graph.undirected(add_inverse=True)
        adj = graph.adjacency
        dense_adj = adj.to_dense().to(torch.int64)
        assert adj.shape == (self.num_nodes, self.num_nodes, self.num_relation)

        # convert dense adjacency matrix of (|V|, |V|, relation_num) to one of (|V|, |V|) with relation specified in matrix
        full_ind = torch.arange(graph.num_relation * 2).repeat(self.num_nodes, self.num_nodes, 1)
        adj_mat = torch.zeros((self.num_nodes, self.num_nodes, 2)) - 1
        adj_mat = adj_mat.to(torch.int64)
        adj_mat = torch.scatter(adj_mat, -1, dense_adj, full_ind)
        adj_mat = adj_mat[:, :, 1]

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()


        adj_mat = adj_mat.unsqueeze(0)
        batched_graphs = self.relation_emb(adj_mat)  # B x N x N x node_dim
        assert batched_graphs.shape == (1, self.num_nodes, self.num_nodes, self.dim)

        # currently consider all pairs of attention, mask is all false; if train graph is not full graph,
        # mask nodes not existent in train graph
        mask = torch.tensor([False] * self.num_nodes).unsqueeze(0)

        if mask is not None:
            new_mask = mask.unsqueeze(2) + mask.unsqueeze(1)
            new_mask = new_mask.unsqueeze(3) + mask.unsqueeze(1).unsqueeze(2)

            mask = new_mask

        if not self.share_layers:
            for mod in self.layers:
                batched_graphs = mod(batched_graphs, mask=mask)
        else:
            for i in range(self.num_message_rounds):
                batched_graphs = self.layers[0](batched_graphs, mask=mask)

        return batched_graphs

    def _reset_parameters(self):

        # for n,p in self.named_parameters():
        #     if ("linear" in n and "weight" in n) or ("embedding" in n):
        #         torch.nn.init.orthogonal_(p)
        #     else:
        #         if p.dim()>1:
        #             nn.init.xavier_uniform_(p)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

@R.register("model.edgeformer")
class EdgeTransformer(nn.Module, core.Configurable):
    def __init__(self, num_message_rounds=8, dropout=0.2, dim=200, num_heads=4, max_grad_norm=1.0, share_layers=True,
                 no_share_layers=False, data_path='', lesion_values=False, lesion_scores=False,  flat_attention=False,
                 ff_factor=4, num_relation=26, num_nodes=104, target_size=25):
        super().__init__()

        self.save_hyperparameters()

        self.num_heads = num_heads
        self.num_relation = num_relation
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.dim = dim
        self.ff_factor = ff_factor
        self.flat_attention = flat_attention

        input_dim = dim
        self.decoder2vocab = get_mlp(
            input_dim,
            target_size
        )

        self.crit = nn.CrossEntropyLoss(reduction='mean')

    def load_train_graph(self, train_set):
        self.train_full = train_set
        h_list = []
        t_list = []
        r_list = []
        for i in range(len(self.train_full)):
            h_list.append(self.train_full[i][0])
            t_list.append(self.train_full[i][1])
            r_list.append(self.train_full[i][2])
        h_list = torch.tensor(h_list)
        t_list = torch.tensor(t_list)
        r_list = torch.tensor(r_list)
        self.encoder = EdgeTransformerEncoder(h_list, t_list, r_list, self.num_heads, self.num_relation, self.num_nodes,
                                              self.dropout, self.dim, self.ff_factor, self.flat_attention)


    def configure_optimizers(self):
        # We will support Adam or AdamW as optimizers.
        if self.model_config.optimizer == "AdamW":
            opt = AdamW
        elif self.model_config.optimizer == "Adam":
            opt = Adam
        optimizer = opt(self.parameters(), **self.model_config.optimizer_args)

        return {'optimizer': optimizer}

        # return {'optimizer':optimizer}

    def _calculate_loss(self, batch):
        batched_graphs = self.encoder(batch)

        query_edges = batch['query_edges']

        logits = self.decoder2vocab(batched_graphs[query_edges[:, 0], query_edges[:, 1], query_edges[:, 2]])

        loss = self.crit(logits, batch['query_labels'])

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch)

        scheduler = self.lr_schedulers()

        return loss

    def compute_acc(self, batch, scores):
        preds = scores.max(-1)[1]

        labels = batch['query_labels']

        acc = ((torch.eq(preds, labels).sum(0)) / preds.size(0)).detach()

        return acc

    def validation_step(self, batch, batch_idx):
        loss, logits = self._calculate_loss(batch)

        acc = self.compute_acc(batch, logits)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits = self._calculate_loss(batch)

        acc = self.compute_acc(batch, logits)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)