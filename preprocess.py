#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time: 2020/9/14
    @Author: menghuanlater
    @Software: Pycharm 2019.2
    @Usage: data preprocess
-----------------------------
    Description:
-----------------------------
"""
import json
import os
import pickle
from random import shuffle
import numpy as np

train_file = open("DataSet/round1_train_0907.json", "r", encoding="UTF-8")
test_file = open("DataSet/juesai_1011.json", "r", encoding="UTF-8")
train_data = json.load(train_file)

x = []

for item in train_data:
    for jtem in item["annotations"]:
        x.append({
            "context": item["text"],
            "query": jtem["Q"],
            "answer": jtem["A"]
        })

for i in range(10):
    shuffle(x)

output = {
    "train_items": x[3000:],
    "test_items": list(json.load(test_file)),
    "valid_items": x[:3000],
    "dureader_train_items": [],
    "cmrc_train_items": [],
    "drcd_train_items": [],
    "multi_task_epoch": 6
}
print("===完成比赛数据处理===")


# 首先处理crmc数据 ==> 相对标准
def cmrc_json(data):
    for dtem in data:
        paragraphs = dtem["paragraphs"]
        for ptem in paragraphs:
            context = ptem["context"][:600]
            qas = ptem["qas"]
            for qtem in qas:
                query = qtem["question"]
                answer = qtem["answers"][0]["text"]
                output["cmrc_train_items"].append({
                    "context": context, "query": query, "answer": answer
                })


for file in os.listdir("DataSet/MultiTask/CMRC"):
    with open("DataSet/MultiTask/CMRC/" + file, "r", encoding="UTF-8") as f:
        cmrc_json(json.load(f)["data"])


print("===完成CMRC数据处理===")

from langconv import *
obj = Converter('zh-hans')


# 其次处理DRCD数据
def drcd_json(data):
    for dtem in data:
        paragraphs = dtem["paragraphs"]
        for ptem in paragraphs:
            context = obj.convert(ptem["context"][:600])
            qas = ptem["qas"]
            for qtem in qas:
                query = obj.convert(qtem["question"])
                answer = obj.convert(qtem["answers"][0]["text"])
                output["drcd_train_items"].append({
                    "context": context, "query": query, "answer": answer
                })


for file in os.listdir("DataSet/MultiTask/DRCD"):
    with open("DataSet/MultiTask/DRCD/" + file, "r", encoding="UTF-8") as f:
        drcd_json(json.load(f)["data"])

print("===完成DRCD数据处理===")


# 最后处理DuReader(完全是用来粗调的==> 粒度太碎)
def dureader_json(data):
    for item in data:
        if item["question_type"] == "YES_NO":
            continue
        context = ""
        for doc_item in item["documents"]:
            if doc_item["is_selected"]:
                context += " ".join(doc_item["paragraphs"])
                if len(context) >= 600:
                    break
        context = context[:600]
        answers = item["answers"]
        for atem in answers:
            output["dureader_train_items"].append({
                "context": context,
                "query": item["question"],
                "answer": atem
            })


for file in os.listdir("DataSet/MultiTask/DuReader/devset"):
    with open("DataSet/MultiTask/DuReader/devset/" + file, "r", encoding="UTF-8") as f:
        dureader_json([json.loads(s) for s in f.readlines()])
print("===完成DuReader Dev数据处理===")

for file in os.listdir("DataSet/MultiTask/DuReader/trainset"):
    with open("DataSet/MultiTask/DuReader/trainset/" + file, "r", encoding="UTF-8") as f:
        dureader_json([json.loads(s) for s in f.readlines()])
print("===完成DuReader Train数据处理===")

for i in range(3):
    shuffle(output["dureader_train_items"])
    shuffle(output["drcd_train_items"])
    shuffle(output["cmrc_train_items"])
print("CMRC2018用于训练的数据一共有%d条" % len(output["cmrc_train_items"]))
print("DRCD用于训练的数据一共有%d条" % len(output["drcd_train_items"]))
print("DuReader用于训练的数据一共有%d条" % len(output["dureader_train_items"]))
with open("DataSet/multi_task.pkl", "wb") as f:
    pickle.dump(output, f)

