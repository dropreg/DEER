import torch
import json
import argparse
import os


def handle_state_dict(xlmr_model_file, save_model_file, index_select):
    # please replace the llama_path with real path
    with open(xlmr_model_file, "rb") as f:
        xlmr_state = torch.load(f, map_location=torch.device("cpu"))
    
    index_select.append(xlmr_state['model']['decoder.sentence_encoder.embed_tokens.weight'].size(0) - 1)
    xlmr_state['model']['decoder.sentence_encoder.embed_tokens.weight'] = \
    xlmr_state['model']['decoder.sentence_encoder.embed_tokens.weight'].index_select(dim=0, index=torch.LongTensor(index_select))
    
    xlmr_state['model']['decoder.lm_head.weight'] = \
    xlmr_state['model']['decoder.lm_head.weight'].index_select(dim=0, index=torch.LongTensor(index_select))

    xlmr_state['model']['decoder.lm_head.bias'] = \
    xlmr_state['model']['decoder.lm_head.bias'].index_select(dim=0, index=torch.LongTensor(index_select))
    
    torch.save(xlmr_state, save_model_file)
    print("dump new model to {}".format(save_model_file))

def extract_dict(train_file_dir):

    token_set = set()
    # for file_item in ["train.spm.de", 'train.spm.en', 'valid.spm.de', 'valid.spm.en', 'test.spm.de', 'test.spm.en']:
    for file_item in ["train.spm.ro", 'train.spm.en', 'valid.spm.ro', 'valid.spm.en', 'test.spm.ro', 'test.spm.en']:
        for lines in open(train_file_dir + file_item).readlines():
            token_list = lines.strip().split()
            for token in token_list:
                token_set.add(token)
    
    offset = 4
    index_select = [0, 1, 2, 3]
    xlmr_idx = 0
    with open(os.path.join(train_file_dir, "dict.txt"), 'w') as fw:
        for line in open("/opt/data/private/data/xlmr/xlmr.base/dict.txt").readlines():
            if line.strip().split()[0] in token_set or line.strip().split()[0].strip() in token_set:
                index_select.append(xlmr_idx + offset)
                fw.writelines(line)
            xlmr_idx += 1
    return index_select

def main():

    xlmr_model_file = "/opt/data/private/data/xlmr/xlmr.base/model.pt"
    save_model_file = "/opt/data/private/data/deer/dict_trunc/enro_model/model.pt"
    train_file_dir = "/opt/data/private/data/deer/dict_trunc/enro_data/raw/"
    
    # xlmr_model_file = "/opt/data/private/data/xlmr/xlmr.base/model.pt"
    # save_model_file = "/opt/data/private/data/deer/dict_trunc/iwslt_model/model.pt"
    # train_file_dir = "/opt/data/private/data/deer/dict_trunc/iwslt_data/raw/"
    
    # xlmr_model_file = "/opt/data/private/data/xlmr/xlmr.base/model.pt"
    # save_model_file = "/opt/data/private/data/deer/dict_trunc/ende_model/model.pt"
    # train_file_dir = "/opt/data/private/data/deer/dict_trunc/wmt14_ende/"

    print("extract xlmr dict from {}".format(train_file_dir))
    index_select = extract_dict(train_file_dir)
    print("handle xlmr model from {} to {}".format(xlmr_model_file, save_model_file))
    handle_state_dict(xlmr_model_file, save_model_file, index_select)


if __name__ == "__main__":
    main()
