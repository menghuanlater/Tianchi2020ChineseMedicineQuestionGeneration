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

from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
import json
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
        context_tokens = tokenizer.tokenize(context)
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


class MyModel(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.decoder_layer = XLDecoder(
            dim=args["dimension"], embedding_matrix=self.roberta_encoder.get_input_embeddings(),
            seq_len=args["max_dec_len"])

    def forward(self, input_ids, input_mask, input_seg, decode_input=None, decode_target=None):
        encoder_rep = self.roberta_encoder(input_ids, input_mask, input_seg)[0]
        return self.decoder_layer(input_ids, encoder_rep, input_mask, decode_input, decode_target,
                                  args["use_beam_search"],
                                  args["beam_width"])


class InitializeNetWeight(object):
    def __init__(self, init_range, init_std):
        self.init_range = init_range
        self.init_std = init_std

    def _init_weight(self, weight):
        nn.init.normal_(weight, self.init_range, self.init_std)

    @staticmethod
    def _init_bias(bias):
        nn.init.constant_(bias, 0)

    def _init_emb_proj(self, proj):
        if self.init_method == "normal":
            nn.init.normal_(proj, 0.0, self.proj_init_std)
        elif self.init_method == "uniform":
            nn.init.uniform_(proj, self.init_range, self.proj_init_std)

    def _init_weights(self, m):
        """
        :param parameters:
        """
        classname = m.__class__.__name__
        if classname.find("Embedding") != -1:  # 解码器部分不能动embedding矩阵的参数
            return
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "bias"):
                self._init_bias(m.bias)

    def init_weights(self, model):
        model.apply(self._init_weights)
        print("random initialize weights succeed.")


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
    def __init__(self, train_loader):
        self.train_loader = train_loader
        self.model = MyModel(pre_train_dir=args["pre_train_dir"])

        self.init_obj = InitializeNetWeight(init_std=args["init_std"], init_range=args["init_range"])
        self.init_obj.init_weights(self.model.decoder_layer)
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
        self.model = nn.parallel.DistributedDataParallel(module=self.model, dim=0, find_unused_parameters=True)

    def train(self):
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
                if steps % args["save_interval"] == 0:
                    torch.save(self.model.module.state_dict(), f=args["save_path"])
                if steps >= args["max_steps"]:
                    break
            if steps >= args["max_steps"]:
                break
        writer.flush()
        writer.close()


if __name__ == "__main__":
    device = "cuda"
    args = {
        "init_lr": 2e-5,
        "batch_size": 24,
        "mos": 2,
        "weight_decay": 0.01,
        "warm_up_steps": 3600,
        "lr_decay_steps": 56000,
        "max_steps": 60000,
        "min_lr_rate": 1e-9,
        "save_interval": 1000,
        "save_path": "ModelStorage/xl_dureader.pth",
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
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1, init_method='tcp://localhost:7001')
        writer = SummaryWriter(logdir="RunLog/Multi-DuReader")
        train_dataset = MyDataset(data=x["dureader_train_items"], max_enc_len=args["max_enc_len"],
                                  max_dec_len=args["max_dec_len"])
        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)

        m = Main(train_loader)
        m.train()
    else:
        print("Invalid args.")
