import copy
import math
import torch
from torch import nn
from torch.nn import Parameter, ModuleList, LayerNorm, Dropout
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_sum

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

    def __init__(self, num_heads, num_relation, num_nodes, dropout, dim, ff_factor, share_layers, num_message_rounds, flat_attention, dependent=True, activation="relu", emb_aggregate='mean'):
        super().__init__()

        self.num_heads = num_heads
        self.num_relation = num_relation
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.dim = dim
        self.ff_factor =ff_factor
        self.deep_residual = False
        self.share_layers = share_layers
        self.num_layers = num_message_rounds
        self.emb_aggregate = emb_aggregate
        self.dependent = dependent

        if not self.dependent:
            # 2 * num_relation(relation and inverse relation)
            # share the same relation embedding over queries
            self.relation_emb = torch.nn.Embedding(num_embeddings=2*self.num_relation, embedding_dim=self.dim)
        else:
            # query specific relation embeddings
            self.query_emb = torch.nn.Embedding(num_embeddings=2*self.num_relation, embedding_dim=self.dim)
            self.relation_linear = torch.nn.Linear(self.dim, 2*self.num_relation.item() * self.dim)

        # 2 * num_relation(relation and inverse relation) + 1(unqueried mask)
        self.mask_emb = torch.nn.Embedding(num_embeddings=2*self.num_relation+1, embedding_dim=self.dim)

        encoder_layer = EdgeTransformerLayer(num_heads, dropout, dim, ff_factor, flat_attention, activation=activation)
        self.layers = _get_clones(encoder_layer, self.num_layers)

        self._reset_parameters()

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
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
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):

        batch_size = h_index.shape[0]
        num_samples = h_index.shape[1]

        init_input = torch.stack([self.mask_emb(torch.tensor(2 * self.num_relation, device=graph.device)).clone().detach() for i in range(self.num_nodes * self.num_nodes)])
        init_input.requires_grad = True
        init_input.to(graph.device)

        # fill in original graph relation embeddings
        # test only

        tid = torch.randint(104, (2, 12216), device=graph.device)
        rid = torch.randint(50, (1, 12216), device=graph.device)
        adj_ind = torch.cat([tid, rid], dim=0)
        '''
        adj = graph.adjacency
        adj_ind = adj._indices()
        '''
        if not self.dependent:
            graph_r_ind = adj_ind[2]
            graph_r_emb = self.relation_emb(graph_r_ind)
            origin_index = adj_ind[0] * self.num_nodes + adj_ind[1]
            origin_index = origin_index.repeat(self.dim, 1).T

            # first set zero before filling
            rec_emb = torch.zeros(origin_index.shape, device=graph.device)
            init_input = init_input.scatter(0, origin_index, rec_emb)

            # then fill in relations and query specific masks
            if self.emb_aggregate == "sum":
                fill_emb = scatter_add(graph_r_emb, origin_index, dim=0, dim_size=self.num_nodes * self.num_nodes)
            elif self.emb_aggregate == "mean":
                fill_emb = scatter_mean(graph_r_emb, origin_index, dim=0, dim_size=self.num_nodes * self.num_nodes)
            elif self.emb_aggregate == "max":
                fill_emb = scatter_max(graph_r_emb, origin_index, dim=0, dim_size=self.num_nodes * self.num_nodes)[0]
            else:
                raise NotImplementedError

            graph_emb = init_input + fill_emb
            assert graph_emb.shape == (self.num_nodes * self.num_nodes, self.dim)

            # creat B batches with same graph input
            # batched_graph_input = graph_emb.repeat(batch_size, 1, 1)
            batched_graph_input = torch.stack([graph_emb.clone().detach() for i in range(batch_size)])
            batched_graph_input.requires_grad = True
            batched_graph_input = batched_graph_input.view(-1, self.dim)
            assert batched_graph_input.shape == (self.num_nodes * self.num_nodes * batch_size, self.dim)

            # add query specific mask to each batch
            batch_ind = torch.arange(batch_size, device=graph.device).repeat(num_samples, 1).T
            assert batch_ind.shape == h_index.shape
            query_mask_ind = batch_ind * self.num_nodes * self.num_nodes + h_index * self.num_nodes + t_index
            query_mask_ind = query_mask_ind.flatten()
            repeat_query_mask_ind = query_mask_ind.repeat(self.dim, 1).T

            r_emb = self.mask_emb(r_index.flatten())
            assert repeat_query_mask_ind.shape == r_emb.shape

            batched_graph_input = batched_graph_input.scatter(0, repeat_query_mask_ind, r_emb)
            final_graph_input = batched_graph_input.view(batch_size, self.num_nodes, self.num_nodes, self.dim)

        else:
            r_batch = r_index[:, 0]
            query = self.query_emb(r_batch)
            assert query.shape[0] == batch_size
            rel_batch_emb = self.relation_linear(query).view(batch_size, 2 * self.num_relation, self.dim)
            rel_batch_emb = rel_batch_emb.view(batch_size * 2 * self.num_relation, self.dim)

            batched_graph_input = torch.stack([init_input.clone().detach() for i in range(batch_size)])
            batched_graph_input = batched_graph_input.view(-1, self.dim)
            assert batched_graph_input.shape == (self.num_nodes * self.num_nodes * batch_size, self.dim)

            origin_index = adj_ind[0] * self.num_nodes + adj_ind[1]
            nnz = len(origin_index)
            origin_index = origin_index.repeat(1, batch_size).squeeze()
            batch_id = torch.arange(batch_size).repeat_interleave(nnz)
            batched_origin_index = batch_id * self.num_nodes * self.num_nodes + origin_index

            graph_r = adj_ind[2]
            graph_r = graph_r.repeat(1, batch_size).squeeze()
            batched_graph_r = batch_id * 2 * self.num_relation + graph_r
            batched_graph_r_emb = rel_batch_emb[batched_graph_r]

            batched_origin_index = batched_origin_index.repeat(self.dim, 1).T

            # first set zero before filling
            rec_emb = torch.zeros(batched_origin_index.shape, device=graph.device)
            batched_graph_input = batched_graph_input.scatter(0, batched_origin_index, rec_emb)

            # then fill in relations and query specific masks
            if self.emb_aggregate == "sum":
                fill_emb = scatter_add(batched_graph_r_emb, batched_origin_index, dim=0, dim_size=batch_size * self.num_nodes * self.num_nodes)
            elif self.emb_aggregate == "mean":
                fill_emb = scatter_mean(batched_graph_r_emb, batched_origin_index, dim=0, dim_size=batch_size * self.num_nodes * self.num_nodes)
            elif self.emb_aggregate == "max":
                fill_emb = scatter_max(batched_graph_r_emb, batched_origin_index, dim=0, dim_size=batch_size * self.num_nodes * self.num_nodes)[0]
            else:
                raise NotImplementedError

            graph_emb = batched_graph_input + fill_emb
            graph_emb.requires_grad = True

            # add query specific mask to each batch
            batch_ind = torch.arange(batch_size, device=graph.device).repeat(num_samples, 1).T
            assert batch_ind.shape == h_index.shape
            query_mask_ind = batch_ind * self.num_nodes * self.num_nodes + h_index * self.num_nodes + t_index
            query_mask_ind = query_mask_ind.flatten()
            repeat_query_mask_ind = query_mask_ind.repeat(self.dim, 1).T

            r_emb = self.mask_emb(r_index.flatten())
            assert repeat_query_mask_ind.shape == r_emb.shape

            graph_emb = graph_emb.scatter(0, repeat_query_mask_ind, r_emb)
            final_graph_input = graph_emb.view(batch_size, self.num_nodes, self.num_nodes, self.dim)


        # currently consider all pairs of attention, mask is all false; if train graph is not full graph,
        # mask nodes not existent in train graph
        mask = torch.tensor([False] * self.num_nodes, device=graph.device).unsqueeze(0)

        if mask is not None:
            new_mask = mask.unsqueeze(2) + mask.unsqueeze(1)
            new_mask = new_mask.unsqueeze(3) + mask.unsqueeze(1).unsqueeze(2)

            mask = new_mask

        if not self.share_layers:
            for mod in self.layers:
                batched_graphs = mod(final_graph_input, mask=mask)
        else:
            for i in range(self.num_layers):
                batched_graphs = self.layers[0](final_graph_input, mask=mask)

        # calculate final representation
        batched_graphs_loss = batched_graphs.view(batch_size * self.num_nodes * self.num_nodes, self.dim)
        binary_rep = batched_graphs_loss[query_mask_ind]
        rel_rep_regularize = self.relation_emb(r_index.flatten())
        assert binary_rep.shape == rel_rep_regularize.shape
        score = torch.sum(binary_rep * rel_rep_regularize, dim=-1)
        #final_rep = torch.cat([binary_rep, rel_rep_regularize], dim=-1)
        #final_rep = final_rep.view(batch_size, -1, 2 * self.dim)
        score = score.view(batch_size, -1)
        return score

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
    def __init__(self, num_message_rounds=8, dropout=0.2, dim=200, num_heads=4, num_mlp_layer=2, remove_one_hop=False, max_grad_norm=1.0, share_layers=True,
                 no_share_layers=False, data_path='', lesion_values=False, lesion_scores=False, flat_attention=False,
                 ff_factor=4, num_relation=26, num_nodes=104, target_size=25, dependent=True):
        super().__init__()

        self.num_heads = num_heads
        self.num_message_rounds = num_message_rounds
        self.num_relation = num_relation
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.dim = dim
        self.ff_factor = ff_factor
        self.flat_attention = flat_attention
        self.num_mlp_layer = num_mlp_layer
        self.share_layers = share_layers
        self.remove_one_hop = remove_one_hop
        self.dependent = dependent

        input_dim = dim
        self.decoder2vocab = get_mlp(
            input_dim,
            target_size
        )

        self.crit = nn.CrossEntropyLoss(reduction='mean')
        self.encoder = EdgeTransformerEncoder(self.num_heads, self.num_relation, self.num_nodes,
                                              self.dropout, self.dim, self.ff_factor, self.share_layers, self.num_message_rounds, self.flat_attention, self.dependent)
        self.mlp = layers.MLP(2 * self.dim, [self.dim] * (self.num_mlp_layer - 1) + [1])

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
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
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        assert graph.num_relation
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        score = self.encoder(graph, h_index, t_index, r_index, all_loss=None, metric=None)
        #score = self.mlp(final_rep)
        #score = score.squeeze(-1)
        assert score.shape == h_index.shape
        return score


