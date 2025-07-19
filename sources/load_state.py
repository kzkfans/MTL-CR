import torch
import torch.nn as nn
import os
from transformers import BartForConditionalGeneration, BartConfig
from models.bart import BartForClassificationAndGeneration, BartForMoe

def load_partial_weights(new_model, old_model):
    """
    将旧模型的部分参数加载到新模型中，跳过修改的部分。
    """
    # 获取旧模型的 state_dict
    old_state_dict = old_model.state_dict()
    # 获取新模型的 state_dict
    new_state_dict = new_model.state_dict()

    # 遍历旧模型的参数
    for name, param in old_state_dict.items():
        # 如果参数在新模型中存在且不需要修改
        if name in new_state_dict and not name.startswith('model.encoder.layers.8'):  # 假设修改了 encoder 的后四层（从第 8 层开始）
            new_state_dict[name].copy_(param)

    # 将更新后的 state_dict 载入新模型
    new_model.load_state_dict(new_state_dict)


def main():
    # 1. 加载旧模型
    config = BartConfig.from_json_file(os.path.join('../pre_trained/models/all', 'config.json'))
    old_model = BartForClassificationAndGeneration.from_pretrained(os.path.join('../pre_trained/models/all', 'pytorch_model.bin'),
                                                               config=config)

    # 2. 定义新模型
    config = old_model.config  # 使用旧模型的配置
    new_model = BartForMoe(config)

    # 3. 加载旧模型的部分参数到新模型
    load_partial_weights(new_model, old_model)

    # 4. 保存新模型
    new_model.save_pretrained('../pre_trained/models/new')
    print("新模型已保存至 '../pre_trained/models/new'")

if __name__ == "__main__":
    main()