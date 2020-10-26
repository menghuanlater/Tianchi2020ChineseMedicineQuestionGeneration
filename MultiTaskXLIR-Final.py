#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020/9/14
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage: data preprocess
-----------------------------
    Description: Base on RoBERTa and Transformer-XL Decoder and Copy Mechanism
    Transformer Decoder采用Transformer-XL
-----------------------------
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2][4:]
from typing import Any

from transformers import BertTokenizer, BertModel, BertConfig
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
import json
import re
from copy import deepcopy
from tensorboardX import SummaryWriter


class MyDataset(Dataset):
    def __init__(self, data, max_enc_len, max_dec_len):
        self.data = data
        self.max_encode_len = max_enc_len
        self.max_decode_len = max_dec_len
        self.SEG_A = 0
        self.SEG_P = 1
        self.ID_PAD = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        context, query, answer = item["context"], item["query"], item["answer"]
        context_tokens = tokenizer.tokenize(context.replace("\n", " ").replace("\t", " ").replace("\\", ""))
        query_tokens = tokenizer.tokenize(query)
        answer_tokens = tokenizer.tokenize(answer)[:args["max_answer_len"]]

        c = ["[CLS]"] + answer_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_encode_len - 1:
            c = c[:self.max_encode_len - 1]
        c += ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [self.SEG_A] * (len(answer_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(answer_tokens))
        extra = self.max_encode_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0.0] * extra
            input_seg += [self.SEG_P] * extra
        if len(query_tokens) > self.max_decode_len - 1:
            query_tokens = query_tokens[: self.max_decode_len - 1]
        c = tokenizer.convert_tokens_to_ids(query_tokens)
        dec_input = [args["start_token_id"]] + c
        dec_target = c + [args["end_token_id"]]
        extra = self.max_decode_len - len(dec_input)
        if extra > 0:
            dec_input += [self.ID_PAD] * extra
            dec_target += [self.ID_PAD] * extra
        return {
            "input_ids": torch.tensor(input_ids).long(), "input_mask": torch.tensor(input_mask).float(),
            "input_seg": torch.tensor(input_seg).long(), "decode_input": torch.tensor(dec_input).long(),
            "decode_target": torch.tensor(dec_target).long(), "label": query
        }


class XLRelPosEmb(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, d_embed):
        super().__init__()

        self.d_embed = d_embed
        inv_freq = 1 / (10000 ** (torch.arange(0.0, self.d_embed, 2.0) / self.d_embed))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb


class PositionwiseFFN(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, d_model, d_inner, layer_norm_epsilon=1e-5):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(p=args["dropout"]),
            nn.Linear(d_inner, d_model),
            nn.Dropout(p=args["dropout"])
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

    def forward(self, inp):
        core_out = self.CoreNet(inp)
        output = self.layer_norm(inp + core_out)
        return output


class RelPartialLearnableMultiHeadAttn(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, n_heads, d_model, layer_norm_epsilon=1e-5):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.mask_attn_qkv_net = nn.Linear(d_model, 3 * d_model, bias=False)
        self.mask_attn_o_net = nn.Linear(d_model, d_model, bias=False)

        self.interaction_kv_net = nn.Linear(d_model, 2 * d_model, bias=False)
        self.interaction_q_net = nn.Linear(d_model, d_model, bias=False)
        self.interaction_o_net = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm_mask_attn = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.layer_norm_interaction = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.scale = 1 / (self.d_head ** 0.5)

        self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_heads, self.d_head))

        self.r_net = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(p=args["dropout"])

    @staticmethod
    def _rel_shift(x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, enc_context, attn_mask, padding_mask):
        # attn_mask用于Masked-Attn Mechanism(decode自身部分)
        # padding_mask用于Norm Multi-Attn, Decode与Encode Contextual Rep交互
        dec_len, bsz, enc_len = w.size(0), w.size(1), enc_context.size(0)
        w_heads = self.mask_attn_qkv_net(w)
        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        w_head_q = w_head_q.view(dec_len, bsz, self.n_heads, self.d_head)  # dec_len x bsz x n_head x d_head
        w_head_k = w_head_k.view(dec_len, bsz, self.n_heads, self.d_head)  # dec_len x bsz x n_head x d_head
        w_head_v = w_head_v.view(dec_len, bsz, self.n_heads, self.d_head)  # dec_len x bsz x n_head x d_head

        r_head_k = r_head_k.view(dec_len, self.n_heads, self.d_head)  # dec_len x n_head x d_head
        rw_head_q = w_head_q + self.r_w_bias  # dec_len x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)  # dec_len x dec_len x bsz x n_head
        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)  # dec_len x dec_len x bsz x n_head
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # causal masking mechanism
        attn_mask = attn_mask == 0  # switch to bool
        attn_score = attn_score.float().masked_fill(attn_mask, -1e30).type_as(attn_score)
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        attn_vec = attn_vec.contiguous().view(dec_len, bsz, self.d_model)

        attn_out = self.mask_attn_o_net(attn_vec)
        attn_out = self.drop(attn_out)

        mask_attn_output = self.layer_norm_mask_attn(w + attn_out)

        # 与编码器交互部分
        inter_k, inter_v = torch.chunk(self.interaction_kv_net(enc_context), 2, dim=-1)
        inter_q = self.interaction_q_net(mask_attn_output)
        inter_q = inter_q.view(dec_len, bsz, self.n_heads, self.d_head)
        inter_k = inter_k.view(enc_len, bsz, self.n_heads, self.d_head)
        inter_v = inter_v.view(enc_len, bsz, self.n_heads, self.d_head)

        attn_score = torch.einsum("qbnd,kbnd->qkbn", inter_q, inter_k)
        attn_score.mul_(self.scale)

        # use padding_mask to mask input [PAD] token
        padding_mask = padding_mask[None, :, :, None].repeat(dec_len, 1, 1, 1)
        attn_score = attn_score + (1 - padding_mask) * (-1e30)
        attn_prob = torch.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, inter_v)
        attn_vec = attn_vec.contiguous().view(dec_len, bsz, self.d_model)

        attn_out = self.interaction_o_net(attn_vec)
        attn_out = self.drop(attn_out)

        interaction_output = self.layer_norm_interaction(attn_out + mask_attn_output)
        return interaction_output


class RelPartialLearnableDecoderLayer(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, n_heads, d_model, d_inner):
        super().__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_heads=n_heads, d_model=d_model)
        self.ffn_layer = PositionwiseFFN(d_model=d_model, d_inner=d_inner)

    def forward(self, dec_inp, r, enc_inp, dec_mask, enc_mask):
        attn_output = self.dec_attn(w=dec_inp, r=r, enc_context=enc_inp, attn_mask=dec_mask, padding_mask=enc_mask)
        ffn_out = self.ffn_layer(attn_output)
        return ffn_out


class XLDecoder(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dim, embedding_matrix: nn.Embedding, seq_len):
        super().__init__()
        self.d_model = dim
        self.word_emb = embedding_matrix
        self.seq_len = seq_len
        self.n_layers = args["decoder_layers"]
        self.n_heads = 16
        self.ffn = 4 * dim
        self.epsilon = 1e-6

        self.drop = nn.Dropout(p=args["dropout"])
        self.pos_emb = XLRelPosEmb(d_embed=dim)
        self.layers = nn.ModuleList()

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_heads=self.n_heads, d_model=self.d_model, d_inner=self.ffn
                )
            )
        self.output = nn.Linear(in_features=dim, out_features=dim)
        self.copy_output = nn.Linear(in_features=dim, out_features=dim)
        # 自适应的解码概率结合
        self.mode_select = nn.Sequential(
            nn.Linear(in_features=dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, encoder_rep, input_mask, decode_input, decode_target, use_beam_search, beam_width):
        bsz = input_ids.size(0)
        if decode_input is not None:  # 代表训练模式
            input_ids = input_ids[:, None, :].repeat(1, self.seq_len, 1)
            decode_embed = self.drop(self.word_emb(decode_input))
            all_ones = decode_embed.new_ones((self.seq_len, self.seq_len), dtype=torch.uint8)
            dec_attn_mask = torch.tril(all_ones, diagonal=0)[:, :, None, None]
            pos_seq = torch.arange(self.seq_len - 1, -1, -1.0, device=device, dtype=decode_embed.dtype)
            pos_embed = self.drop(self.pos_emb(pos_seq))
            core_out = decode_embed.transpose(0, 1).contiguous()
            enc_rep_t = encoder_rep.transpose(0, 1).contiguous()
            enc_mask_t = input_mask.transpose(0, 1).contiguous()
            for layer in self.layers:
                core_out = layer(
                    dec_inp=core_out, r=pos_embed, enc_inp=enc_rep_t,
                    dec_mask=dec_attn_mask, enc_mask=enc_mask_t
                )
            core_out = self.drop(core_out.transpose(0, 1).contiguous())  # (bsz, dec_len, dim)
            output = self.output(core_out)
            vocab_logits = torch.nn.functional.linear(input=output, weight=self.word_emb.weight)
            vocab_prob = torch.softmax(vocab_logits, dim=-1)
            input_logits = torch.einsum("bid,bjd->bij", self.copy_output(core_out), encoder_rep)  # (bsz, dec_len, enc_len)
            input_logits = input_logits + (1.0 - input_mask[:, None, :].repeat(1, self.seq_len, 1)) * (-1e30)
            input_prob = torch.softmax(input_logits, dim=-1)  # (bsz, dec_len, enc_len)
            mode_sig = self.mode_select(core_out)  # (bsz, dec_len, 1)
            vocab_prob = vocab_prob * mode_sig
            vocab_prob = torch.scatter_add(vocab_prob, dim=2, index=input_ids, src=(1 - mode_sig) * input_prob)
            vocab_prob = vocab_prob.view(-1, args["vocab_size"])
            decode_target = decode_target.view(-1)
            predict = torch.gather(vocab_prob, dim=1, index=decode_target[:, None]).squeeze(dim=-1)
            init_loss = -torch.log(predict + self.epsilon)
            init_loss *= (decode_target != 0).float()
            loss = torch.sum(init_loss) / torch.nonzero(decode_target != 0, as_tuple=False).size(0)
            # 为了并行化设计, 将loss变成(bsz,)
            return loss[None].repeat(bsz)
        else:  # 代表验证或者测试解码模式 ==> 比较耗时
            if not use_beam_search:  # 使用贪心搜索 ==> 验证集
                dec_list = []
                decode_ids = torch.full(size=(bsz, 1), fill_value=args["start_token_id"], dtype=torch.int32).long().to(device)
                for i in range(1, self.seq_len + 1):
                    if i > 1:
                        decode_ids = torch.cat([decode_ids, dec_list[i - 2]], dim=-1)
                    decode_embed = self.word_emb(decode_ids)
                    all_ones = decode_embed.new_ones((i, i), dtype=torch.uint8)
                    dec_attn_mask = torch.tril(all_ones, diagonal=0)[:, :, None, None]
                    pos_seq = torch.arange(i - 1, -1, -1.0, device=device, dtype=decode_embed.dtype)
                    pos_embed = self.pos_emb(pos_seq)
                    core_out = decode_embed.transpose(0, 1).contiguous()
                    enc_rep_t = encoder_rep.transpose(0, 1).contiguous()
                    enc_mask_t = input_mask.transpose(0, 1).contiguous()
                    for layer in self.layers:
                        core_out = layer(
                            dec_inp=core_out, r=pos_embed, enc_inp=enc_rep_t,
                            dec_mask=dec_attn_mask, enc_mask=enc_mask_t
                        )
                    core_out = core_out.transpose(0, 1).contiguous()[:, -1, :]
                    output = self.output(core_out)
                    vocab_logits = torch.nn.functional.linear(input=output, weight=self.word_emb.weight)
                    vocab_prob = torch.softmax(vocab_logits, dim=-1)
                    input_logits = torch.einsum("bd,bjd->bj", self.copy_output(core_out), encoder_rep)  # (bsz, enc_len)
                    input_logits = input_logits + (1.0 - input_mask) * (-1e30)
                    input_prob = torch.softmax(input_logits, dim=-1)  # (bsz, enc_len)
                    mode_sig = self.mode_select(core_out)  # (bsz, 1)
                    vocab_prob = vocab_prob * mode_sig
                    vocab_prob = torch.scatter_add(vocab_prob, dim=1, index=input_ids, src=(1 - mode_sig) * input_prob)
                    dec_list.append(torch.argmax(vocab_prob, dim=-1)[:, None])
                return torch.cat(dec_list, dim=-1)
            else:  # 使用集束搜索
                # 扩展成beam_width * bsz
                """
                需要注意: 1. trigram-block的使用 ==> 出现重复直接加上-1e9(需要考虑end_token边界=>只在边界范围内使用)
                2. 长度惩罚, 考虑end_token边界
                """
                decode_ids = torch.full(size=(bsz * beam_width, 1), fill_value=args["start_token_id"], dtype=torch.int32).long().to(device)
                input_ids = input_ids.repeat((beam_width, 1))
                encoder_rep = encoder_rep.repeat((beam_width, 1, 1))
                input_mask = input_mask.repeat((beam_width, 1))
                dec_topK_log_probs = [0] * (beam_width * bsz)  # (bsz*beam)  每个序列的当前log概率和
                dec_topK_sequences = [[] for _ in range(beam_width * bsz)] # (bsz*beam, seq_len) 解码id序列
                dec_topK_seq_lens = [1] * (beam_width * bsz)  # 解码序列长度 ==> 加上一个偏置项1, 防止进行长度惩罚时出现div 0的情况
                for i in range(1, self.seq_len + 1):
                    if i > 1:
                        input_decode_ids = torch.cat([decode_ids, torch.tensor(dec_topK_sequences).long().to(device)], dim=-1)
                    else:
                        input_decode_ids = decode_ids
                    decode_embed = self.word_emb(input_decode_ids)
                    all_ones = decode_embed.new_ones((i, i), dtype=torch.uint8)
                    dec_attn_mask = torch.tril(all_ones, diagonal=0)[:, :, None, None]
                    pos_seq = torch.arange(i - 1, -1, -1.0, device=device, dtype=decode_embed.dtype)
                    pos_embed = self.pos_emb(pos_seq)
                    core_out = decode_embed.transpose(0, 1).contiguous()
                    enc_rep_t = encoder_rep.transpose(0, 1).contiguous()
                    enc_mask_t = input_mask.transpose(0, 1).contiguous()
                    for layer in self.layers:
                        core_out = layer(
                            dec_inp=core_out, r=pos_embed, enc_inp=enc_rep_t,
                            dec_mask=dec_attn_mask, enc_mask=enc_mask_t
                        )
                    core_out = core_out.transpose(0, 1).contiguous()[:, -1, :]
                    output = self.output(core_out)
                    vocab_logits = torch.nn.functional.linear(input=output, weight=self.word_emb.weight)
                    vocab_prob = torch.softmax(vocab_logits, dim=-1)
                    input_logits = torch.einsum("bd,bjd->bj", self.copy_output(core_out), encoder_rep)  # (bsz*beam, enc_len)
                    input_logits = input_logits + (1.0 - input_mask) * (-1e30)
                    input_prob = torch.softmax(input_logits, dim=-1)  # (bsz*beam, enc_len)
                    mode_sig = self.mode_select(core_out)  # (bsz*beam, 1)
                    vocab_prob = vocab_prob * mode_sig
                    vocab_prob = torch.scatter_add(vocab_prob, dim=1, index=input_ids, src=(1 - mode_sig) * input_prob)  # (bsz*beam, vocab)
                    vocab_logp = torch.log(vocab_prob + self.epsilon)  # 取对数， 加eps
                    """ step1: 检查是否存在trigram blocking重叠, 只需要检查最后一项和之前项是否存在重叠即可 """
                    if i > 4:  # 当序列长度大于等于4时才有意义, 或者当前解码时刻大于4时才有检查的必要
                        for j in range(beam_width * bsz):
                            trigram_blocks = []
                            for k in range(3, i):
                                if dec_topK_sequences[j][k-1] == args["end_token_id"]:
                                    break
                                trigram_blocks.append(dec_topK_sequences[j][k-3:k])
                            if len(trigram_blocks) > 1 and trigram_blocks[-1] in trigram_blocks[:-1]:
                                dec_topK_log_probs[j] += -1e9
                    """ step2: 为每个样本, 选择topK个序列 ==> 类似于重构dec_topK_sequences"""
                    for j in range(bsz):
                        topK_vocab_logp = vocab_logp[j::bsz]  # (k, vocab)
                        candidate_list = []
                        """ 容易出错的地方, i=1的时候不需要为每个K生成K个候选,否则beam search将完全沦为greedy search """
                        for k in range(beam_width):
                            ind = j + k * bsz
                            if args["end_token_id"] in dec_topK_sequences[ind]:
                                candidate_list.append({
                                    "add_logit": 0, "add_seq_len": 0, "affiliate_k": k, "add_token_id": args["end_token_id"],
                                    "sort_logits": dec_topK_log_probs[ind] / (dec_topK_seq_lens[ind] ** args["beam_length_penalty"])
                                })
                            else:
                                k_logps, k_indices = topK_vocab_logp[k].topk(dim=0, k=beam_width)
                                k_logps, k_indices = k_logps.cpu().numpy(), k_indices.cpu().numpy()
                                for l in range(beam_width):
                                    aff = l if i == 1 else k
                                    candidate_list.append({
                                        "add_logit": k_logps[l], "add_seq_len": 1, "affiliate_k": aff, "add_token_id": k_indices[l],
                                        "sort_logits": (dec_topK_log_probs[ind] + k_logps[l]) / ((dec_topK_seq_lens[ind] + 1) ** args["beam_length_penalty"])
                                    })
                            if i == 1:  ## 当解码第一个词的时候只能考虑一个
                                break
                        candidate_list.sort(key=lambda x: x["sort_logits"], reverse=True)
                        candidate_list = candidate_list[:beam_width]
                        """ 序列修正, 更新topK """
                        c_dec_topK_sequences, c_dec_topK_log_probs, c_dec_topK_seq_lens = \
                            deepcopy(dec_topK_sequences), deepcopy(dec_topK_log_probs), deepcopy(dec_topK_seq_lens)
                        for k in range(beam_width):
                            ind = bsz * candidate_list[k]["affiliate_k"] + j
                            r_ind = bsz * k + j
                            father_seq, father_logits, father_len = c_dec_topK_sequences[ind], c_dec_topK_log_probs[ind], c_dec_topK_seq_lens[ind]
                            dec_topK_sequences[r_ind] = father_seq + [candidate_list[k]["add_token_id"]]
                            dec_topK_log_probs[r_ind] = father_logits + candidate_list[k]["add_logit"]
                            dec_topK_seq_lens[r_ind] = father_len + candidate_list[k]["add_seq_len"]
                return torch.tensor(dec_topK_sequences[:bsz]).long().to(device)


class MyModel(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str):
        super().__init__()
        self.roberta_encoder = BertModel(config=BertConfig.from_json_file(pre_train_dir+ "config.json"))
        self.decoder_layer = XLDecoder(
            dim=args["dimension"], embedding_matrix=self.roberta_encoder.get_input_embeddings(),
            seq_len=args["max_dec_len"])

    def forward(self, input_ids, input_mask, input_seg, decode_input=None, decode_target=None):
        encoder_rep = self.roberta_encoder(input_ids, input_mask, input_seg)[0]
        return self.decoder_layer(input_ids, encoder_rep, input_mask, decode_input, decode_target,
                                  args["use_beam_search"],
                                  args["beam_width"])


class WarmUp_LinearDecay:
    def __init__(self, optimizer: optim.AdamW, init_rate, warm_up_steps, decay_steps, min_lr_rate):
        self.optimizer = optimizer
        self.init_rate = init_rate
        self.warm_up_steps = warm_up_steps
        self.decay_steps = decay_steps
        self.min_lr_rate = min_lr_rate
        self.optimizer_step = 0

    def step(self):
        self.optimizer_step += 1
        if self.optimizer_step <= self.warm_up_steps:
            rate = (self.optimizer_step / self.warm_up_steps) * self.init_rate
        elif self.warm_up_steps < self.optimizer_step <= (self.warm_up_steps + self.decay_steps):
            rate = (1.0 - ((self.optimizer_step - self.warm_up_steps) / self.decay_steps)) * self.init_rate
        else:
            rate = self.min_lr_rate
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()


class Main(object):
    def __init__(self, train_loader, valid_loader, test_flag=False, test_items=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_items = test_items
        self.model = MyModel(pre_train_dir=args["pre_train_dir"])

        if test_flag:
            self.model.load_state_dict(torch.load(args["save_path"], map_location=device), strict=False)
        else:
            self.model.load_state_dict(torch.load(args["load_path"], map_location=device), strict=False)
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': args["weight_decay"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]

            self.optimizer = optim.AdamW(params=optimizer_grouped_parameters, lr=args["init_lr"])
            self.schedule = WarmUp_LinearDecay(optimizer=self.optimizer, init_rate=args["init_lr"],
                                               warm_up_steps=args["warm_up_steps"],
                                               decay_steps=args["lr_decay_steps"], min_lr_rate=args["min_lr_rate"])
        self.model.to(device=device)
        if args["is_train"]:
            self.model = nn.parallel.DistributedDataParallel(module=self.model, dim=0, find_unused_parameters=True)

    def train(self):
        best_rl = 0.0
        self.model.train()
        steps = 0
        while True:
            for item in self.train_loader:
                input_ids, input_mask, input_seg, decode_input, decode_target = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["decode_input"], item[
                        "decode_target"]
                self.optimizer.zero_grad()
                loss = self.model(
                    input_ids=input_ids.to(device),
                    input_mask=input_mask.to(device),
                    input_seg=input_seg.to(device),
                    decode_input=decode_input.to(device),
                    decode_target=decode_target.to(device)
                )
                loss = loss.float().mean().type_as(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args["clip_norm"])
                self.schedule.step()
                steps += 1
                writer.add_scalar("loss", loss.item(), global_step=steps)
                if steps % args["eval_interval"] == 0:
                    rl = self.valid()
                    writer.add_scalar("valid_rl", rl, global_step=steps)
                    if rl > best_rl:
                        best_rl = rl
                        torch.save(self.model.module.state_dict(), f=args["save_path"])
                if steps >= args["max_steps"]:
                    break
            if steps >= args["max_steps"]:
                break
        writer.flush()
        writer.close()

    def valid(self):
        self.model.eval()
        rouge_l = []
        with torch.no_grad():
            for item in self.valid_loader:
                input_ids, input_mask, input_seg, label = item["input_ids"], item["input_mask"], item["input_seg"], \
                                                          item["label"]
                dec_seq = self.model(
                    input_ids=input_ids.to(device),
                    input_mask=input_mask.to(device),
                    input_seg=input_seg.to(device)
                )
                dec_seq = dec_seq.cpu().numpy()
                for i in range(len(dec_seq)):
                    x = dec_seq[i]
                    s = []
                    for j in x:
                        if int(j) == args["end_token_id"]:
                            break
                        else:
                            s.append(int(j))
                    s = "".join(tokenizer.convert_ids_to_tokens(s))
                    s = s.replace("，", "").replace("[UNK]", "")
                    char_lis = []
                    for c in s:
                        if c not in char_lis:
                            char_lis.append(c)
                    for c in char_lis:
                        try:
                            p = re.compile("(%s){2,}" % c)
                            s = re.sub(p, c, s)
                        except Exception as e:
                            continue
                    rouge_l.append(self.rouge_l(hypo=s, refer=label[i]))
        self.model.train()
        return np.average(rouge_l)

    @staticmethod
    def test_encode(context, answer):
        context_tokens = tokenizer.tokenize(context.replace("\n", " ").replace("\t", " ").replace("\\", ""))
        answer_tokens = tokenizer.tokenize(answer)[:args["max_answer_len"]]
        c = ["[CLS]"] + answer_tokens + ["[SEP]"] + context_tokens
        if len(c) > args["max_enc_len"] - 1:
            c = c[:args["max_enc_len"] - 1]
        c += ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [0] * (len(answer_tokens) + 2) + [1] * (len(input_ids) - 2 - len(answer_tokens))
        extra = args["max_enc_len"] - len(input_ids)
        if extra > 0:
            input_ids += [0] * extra
            input_mask += [0.0] * extra
            input_seg += [1] * extra
        return {
            "input_ids": torch.tensor(input_ids).long().unsqueeze(dim=0).to(device),
            "input_mask": torch.tensor(input_mask).float().unsqueeze(dim=0).to(device),
            "input_seg": torch.tensor(input_seg).long().unsqueeze(dim=0).to(device)
        }

    def test(self):
        self.model.eval()
        output = x["test_items"]
        with torch.no_grad():
            for i in range(len(output)):
                text = output[i]["text"]
                annotations = output[i]["annotations"]
                tmp_enc_ids, tmp_enc_mask, tmp_enc_seg = [], [], []
                for j in range(len(annotations)):
                    y = self.test_encode(text, annotations[j]["A"])
                    tmp_enc_ids.append(y["input_ids"])
                    tmp_enc_mask.append(y["input_mask"])
                    tmp_enc_seg.append(y["input_seg"])
                dec_seq = self.model(
                    input_ids=torch.cat(tmp_enc_ids, dim=0),
                    input_mask=torch.cat(tmp_enc_mask, dim=0),
                    input_seg=torch.cat(tmp_enc_seg, dim=0)
                )
                dec_seq = dec_seq.cpu().numpy()
                for j in range(len(dec_seq)):
                    y = dec_seq[j]
                    s = []
                    for k in y:
                        if int(k) == args["end_token_id"]:
                            break
                        else:
                            s.append(int(k))
                    s = "".join(tokenizer.convert_ids_to_tokens(s))
                    s = s.replace("，", "").replace("[UNK]", "").replace("#", "")
                    char_lis = []
                    for c in s:
                        if c not in char_lis:
                            char_lis.append(c)
                    for c in char_lis:
                        try:
                            p = re.compile("(%s){2,}" % c)
                            s = re.sub(p, c, s)
                        except Exception as e:
                            continue
                    # 针对英文的一些修正
                    t_text = text.lower()
                    p = re.compile("([A-Za-z]+)")
                    m = re.finditer(p, s)
                    for i_match in m:
                        start, end, i_str = i_match.start(), i_match.end(), i_match.group()
                        if i_str in t_text:
                            i_index = t_text.index(i_str)
                            s = s[:start] + text[i_index: i_index + (end - start)] + s[end:]
                    if len(s) < 2:
                        s = annotations[j]["A"]
                    annotations[j]["Q"] = s
                if i % 50 == 0 and i > 0:
                    print("The program has completed %s predictions" % i)
        with open("submit.json", "w", encoding="UTF-8") as f:
            json.dump(output, f, ensure_ascii=False)
            print("The program has completed all predictions")

    @staticmethod
    def rouge_l(hypo, refer):
        if len(hypo) == 0 or len(refer) == 0:
            return 0
        x = [[0 for _ in range(len(refer) + 1)] for _ in range(len(hypo) + 1)]
        lcs = 0
        for i in range(len(hypo)):
            for j in range(len(refer)):
                if hypo[i] == refer[j]:
                    x[i + 1][j + 1] = x[i][j] + 1
                    if x[i + 1][j + 1] > lcs:
                        lcs = x[i + 1][j + 1]
                else:
                    x[i + 1][j + 1] = max(x[i + 1][j], x[i][j + 1])
        p, r = lcs / len(hypo), lcs / len(refer)
        if (p + r) == 0:
            return 0
        else:
            return (2 * p * r) / (p + r)


if __name__ == "__main__":
    device = "cuda"
    args = {
        "init_lr": 2e-5,
        "batch_size": 8,
        "mos": 2,
        "weight_decay": 0.01,
        "warm_up_steps": 1000,
        "lr_decay_steps": 15000,
        "max_steps": 16000,
        "min_lr_rate": 1e-9,
        "eval_interval": 1000,
        "save_path": "ModelStorage/final.pth",
        "load_path": "ModelStorage/xl_dureader_drmc.pth",
        "pre_train_dir": "/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/",
        "clip_norm": 0.25,
        "start_token": "[unused1]",
        "end_token": "[unused2]",
        "start_token_id": 1,
        "end_token_id": 2,
        "dimension": 1024,
        "max_enc_len": 512,
        "max_dec_len": 50,
        "max_answer_len": 100,
        "use_beam_search": False,
        "beam_width": 5,
        "beam_length_penalty": 0.6,
        "decoder_layers": 3,
        "dropout": 0.1,
        "vocab_size": 21128,
        "init_range": 0.02,
        "init_std": 0.02
    }

    with open("DataSet/multi_task.pkl", "rb") as f:
        x = pickle.load(f)

    tokenizer = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")

    if sys.argv[1] == "train":
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1, init_method='tcp://localhost:7011')
        args["is_train"] = True
        writer = SummaryWriter(logdir="RunLog/%s" % sys.argv[3])
        if y is None:
            train_dataset = MyDataset(data=x["train_items"], max_enc_len=args["max_enc_len"],
                                      max_dec_len=args["max_dec_len"])
        else:
            train_dataset = MyDataset(data=x["train_items"] + y, max_enc_len=args["max_enc_len"],
                                      max_dec_len=args["max_dec_len"])
        valid_dataset = MyDataset(data=x["valid_items"], max_enc_len=args["max_enc_len"],
                                  max_dec_len=args["max_dec_len"])

        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4)

        m = Main(train_loader, valid_loader)
        m.train()
    else:
        writer = None
        args["is_train"] = False
        args["use_beam_search"] = True
        m = Main(None, None, test_flag=True, test_items=x["test_items"])
        m.test()
