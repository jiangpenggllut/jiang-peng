# encoding=utf-8

import argparse
from others.logging import init_logger
from prepro import json_to_data as data_builder
import os
#os.environ['CURL_CA_BUNDLE'] = ''
# 离线状态下可运行
#TRANSFORMERS_OFFLINE=1


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-raw_path", default='raw_data/mrpc', type=str)
    parser.add_argument("-save_path", default='data', type=str)
    parser.add_argument("-n_cpus", default=4, type=int)
    parser.add_argument("-random_seed", default=666, type=int)

    # json_to_data args
    parser.add_argument('-num_class', default=2, type=int)
    parser.add_argument('-log_file', default='logs/json_to_data.log')
    parser.add_argument("-tokenizer", default="")
    parser.add_argument('-min_length', default=1, type=int)
    parser.add_argument('-max_length', default=150, type=int)
    parser.add_argument("-truncated", nargs='?', const=True, default=True)
    parser.add_argument("-shard_size", default=5000, type=int)

    args = parser.parse_args()
    init_logger(args.log_file)

    model_names_english = [
        './roberta-base',
        './bert-base-uncased',
        './albert-base-v2',
        './microsoft-deberta-base',
        './funnel-transformer-medium',
        './huawei-noah-EntityCS-39-PEP_MS_MLM-xlmr-base',
        './ngwlh-KBioXLM',
        './Twitter-twhin-bert-base',
        './phjhk-hklegal-xlm-r-base-t',
    ]
    
    model_names_chinese = [
        './hongyin-bert-chinese-homie-base',
        './ngwlh-KBioXLM',
        './Twitter-twhin-bert-base',
        './huawei-noah-EntityCS-39-PEP_MS_MLM-xlmr-base',
        './hfl-chinese-macbert-base',
        './hfl-chinese-roberta-wwm-ext',
        './phjhk-hklegal-xlm-r-base-t',
    ]
    data_name = args.raw_path.split('/')[-1]
    if data_name == 'medical' or data_name == 'kuake-qtr':
        model_names = model_names_chinese
    else:
        model_names = model_names_english
    data_saved_path = args.save_path
    for raw_name in model_names:
        name = raw_name.replace('/', '-')
        args.save_path = data_saved_path + '/' + data_name + '/' + name
        args.tokenizer = raw_name
        data_builder.format_json_to_data(args)
