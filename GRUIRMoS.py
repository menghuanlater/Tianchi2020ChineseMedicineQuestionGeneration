#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020/9/14
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage: data preprocess
-----------------------------
    Description: Base on RoBERTa and GRU
    -- 增加输入token增强机制(输入的token在解码时具有更高的接受概率) copy mechanism
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


class GRUAttnDecoder(torch.nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, dim, embedding_matrix: nn.Embedding, seq_len):
        # 为了保持一致性, context_vector input_vector 以及 hidden_vector保持相同维度
        # 同时为了减少参数, 注意力机制采取点积缩放形式
        super().__init__()
        self.embedding_matrix = embedding_matrix
        self.seq_len = seq_len  # 解码长度
        self.scale = 1 / np.sqrt(dim)
        self.reset_gate = nn.Sequential(
            nn.Linear(in_features=3 * dim, out_features=dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(in_features=3 * dim, out_features=dim),
            nn.Sigmoid()
        )
        self.update = nn.Sequential(
            nn.Linear(in_features=3 * dim, out_features=dim),
            nn.Tanh()
        )
        # self.pi_mos = nn.Sequential(
        #     nn.Linear(in_features=dim, out_features=args["mos"]),
        #     nn.Softmax()
        # )
        # self.output = nn.ModuleList()
        # for i in range(args["mos"]):
        #     self.output.append(nn.Linear(in_features=dim, out_features=dim))
        self.output = nn.Linear(in_features=dim, out_features=dim)
        self.copy_output = nn.Linear(in_features=dim, out_features=dim)
        self.init_hidden_unit = nn.Parameter(torch.FloatTensor(1, dim))  # 状态初始值

        # 自适应的概率结合
        self.mode_select = nn.Sequential(
            nn.Linear(in_features=dim, out_features=1),
            nn.Sigmoid()
        )
        self.epsilon = 1e-6

    def forward(self, input_ids, input_context, context_mask, decode_input, decode_target, use_beam_search, beam_width):
        """
        :param input_ids: 用于解码增强的输入ids序列
        :param input_context: 编码的context (bsz, enc_seq, dim)
        :param context_mask: 沿用encoder部分的input_mask, 将pad的输入忽略
        :param decode_input: 解码输入 ==> 训练时才有
        :param decode_target: 解码目标 ==> 训练时才有, 测试时为空
        :param use_beam_search: 是否启动beam search解码
        :param beam_width: beam宽度
        :return: 训练时返回损失, 测试时返回解码序列
        """
        bsz = input_context.size(0)
        net_state = self.init_hidden_unit.repeat(bsz, 1)
        if decode_target is not None:
            dec_list = []
            decode_emb = self.embedding_matrix(decode_input)  # 作为输入的一部分(bsz, dec_seq, dim)
            for i in range(self.seq_len):
                # step1: 通过注意力机制获取当前的context_rep
                attn_score = torch.einsum("bsd,bd->bs", input_context, net_state)
                attn_score.mul_(self.scale)
                attn_score += (1.0 - context_mask) * (-1e30)
                attn_prob = torch.softmax(attn_score, dim=-1)
                attn_vec = torch.einsum("bs,bsd->bd", attn_prob, input_context)
                # step2: 更新状态
                x = torch.cat([attn_vec, decode_emb[:, i, :], net_state], dim=-1)
                reset_sig = self.reset_gate(x)
                update_sig = self.update_gate(x)
                update_value = self.update(torch.cat([attn_vec, decode_emb[:, i, :], reset_sig * net_state], dim=-1))
                net_state = (1 - update_sig) * net_state + update_sig * update_value
                # step3: 计算分布概率--> mos
                vocab_prob_list = []
                # pi_k = self.pi_mos(net_state)
                # for k in range(args["mos"]):
                #     output = self.output[k](net_state)
                #     vocab_logits = torch.nn.functional.linear(input=output, weight=self.embedding_matrix.weight)
                #     vocab_prob_list.append(torch.softmax(vocab_logits, dim=-1)[..., None])
                # vocab_prob = torch.einsum("bk,bvk->bv", pi_k, torch.cat(vocab_prob_list, dim=-1))
                output = self.output(net_state)
                vocab_logits = torch.nn.functional.linear(input=output, weight=self.embedding_matrix.weight)
                vocab_prob = torch.softmax(vocab_logits, dim=-1)
                input_logits = torch.einsum("bd,bsd->bs", self.copy_output(net_state), input_context)
                input_logits += (1.0 - context_mask) * (-1e30)
                input_prob = torch.softmax(input_logits, dim=-1)  # (bsz, enc_seq)
                # step4: 根据mode_sig混合两个概率
                mode_sig = self.mode_select(net_state)
                vocab_prob = vocab_prob * mode_sig
                vocab_prob = torch.scatter_add(vocab_prob, dim=1, index=input_ids, src=input_prob * (1 - mode_sig))
                dec_list.append(vocab_prob[:, None, :])
            # 计算损失
            predict = torch.cat(dec_list, dim=1)  # (bsz, dec_seq, vocab)
            predict = predict.view(size=(-1, predict.size(-1)))
            decode_target = decode_target.view(size=(-1,))
            predict = torch.gather(predict, dim=1, index=decode_target[:, None]).squeeze(dim=-1)
            init_loss = -torch.log(predict + self.epsilon)
            init_loss *= (decode_target != 0).float()
            loss = torch.sum(init_loss) / torch.nonzero(decode_target != 0, as_tuple=False).size(0)
            return loss[None].repeat(bsz)
        else:
            if use_beam_search:
                pass
            else:  # 贪婪式解码
                dec_list = []
                for i in range(self.seq_len):
                    # step1: 通过注意力机制获取当前的context_rep
                    attn_score = torch.einsum("bsd,bd->bs", input_context, net_state)
                    attn_score.mul_(self.scale)
                    attn_score += (1.0 - context_mask) * (-1e30)
                    attn_prob = torch.softmax(attn_score, dim=-1)
                    attn_vec = torch.einsum("bs,bsd->bd", attn_prob, input_context)
                    # step2: 更新状态
                    if i == 0:
                        emb = self.embedding_matrix(
                            torch.full(size=(bsz,), fill_value=args["start_token_id"], dtype=torch.int32).long().to(device))
                    else:
                        emb = self.embedding_matrix(dec_list[i - 1].squeeze(dim=-1))
                    x = torch.cat([attn_vec, emb, net_state], dim=-1)
                    reset_sig = self.reset_gate(x)
                    update_sig = self.update_gate(x)
                    update_value = self.update(torch.cat([attn_vec, emb, reset_sig * net_state], dim=-1))
                    net_state = (1 - update_sig) * net_state + update_sig * update_value
                    # step3: 计算分布得分
                    vocab_prob_list = []
                    # pi_k = self.pi_mos(net_state)
                    # for k in range(args["mos"]):
                    #     output = self.output[k](net_state)
                    #     vocab_logits = torch.nn.functional.linear(input=output, weight=self.embedding_matrix.weight)
                    #     vocab_prob_list.append(torch.softmax(vocab_logits, dim=-1)[..., None])
                    # vocab_prob = torch.einsum("bk,bvk->bv", pi_k, torch.cat(vocab_prob_list, dim=-1))
                    output = self.output(net_state)
                    vocab_logits = torch.nn.functional.linear(input=output, weight=self.embedding_matrix.weight)
                    vocab_prob = torch.softmax(vocab_logits, dim=-1)
                    input_logits = torch.einsum("bd,bsd->bs", self.copy_output(net_state), input_context)
                    input_logits += (1.0 - context_mask) * (-1e30)
                    input_prob = torch.softmax(input_logits, dim=-1)  # (bsz, enc_seq)
                    # step4: 根据mode_sig混合两个概率
                    mode_sig = self.mode_select(net_state)
                    vocab_prob = vocab_prob * mode_sig
                    vocab_prob = torch.scatter_add(vocab_prob, dim=1, index=input_ids, src=input_prob * (1 - mode_sig))
                    dec_list.append(torch.argmax(vocab_prob, dim=-1)[:, None])
                return torch.cat(dec_list, dim=-1)


class MyModel(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, pre_train_dir: str):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.decoder_cell = GRUAttnDecoder(dim=args["dimension"],
                                           embedding_matrix=self.roberta_encoder.get_input_embeddings(),
                                           seq_len=args["max_dec_len"])
        if args["freeze_roberta"]:
            for p in self.roberta_encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, input_mask, input_seg, decode_input=None, decode_target=None):
        encoder_rep = self.roberta_encoder(input_ids, input_mask, input_seg)[0]
        return self.decoder_cell(input_ids, encoder_rep, input_mask, decode_input, decode_target, args["use_beam_search"],
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
            if args["warm_start"]:
                self.model.load_state_dict(torch.load(args["save_path"], map_location=device), strict=False)
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
                    item["input_ids"], item["input_mask"], item["input_seg"], item["decode_input"], item["decode_target"]
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
                input_ids, input_mask, input_seg, label = item["input_ids"], item["input_mask"], item["input_seg"], item["label"]
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
        context_tokens = tokenizer.tokenize(context)
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
        output = x["test_items"]
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
        "batch_size": 10,
        "weight_decay": 0.01,
        "warm_up_steps": 1000,
        "lr_decay_steps": 15000,
        "max_steps": 18000,
        "min_lr_rate": 1e-9,
        "eval_interval": 1000,
        "save_path": "ModelStorage/gru_ir.pth",
        "mos": 4,
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
        "warm_start": False,
        "freeze_roberta": False
    }

    with open("DataSet/baseline.pkl", "rb") as f:
        x = pickle.load(f)

    tokenizer = BertTokenizer(vocab_file="/home/ldmc/quanlin/Pretrained_NLP_Models/Pytorch/RoBERTa_Large_ZH/vocab.txt")

    if sys.argv[1] == "train":
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1, init_method='tcp://localhost:6666')
        args["is_train"] = True
        writer = SummaryWriter(logdir="RunLog/%s" % sys.argv[3])
        train_dataset = MyDataset(data=x["train_items"], max_enc_len=args["max_enc_len"],
                                  max_dec_len=args["max_dec_len"])
        valid_dataset = MyDataset(data=x["valid_items"], max_enc_len=args["max_enc_len"],
                                  max_dec_len=args["max_dec_len"])

        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4)

        m = Main(train_loader, valid_loader)
        m.train()
    else:
        args["is_train"] = False
        writer = None
        m = Main(None, None, test_flag=True, test_items=x["test_items"])
        m.test()
