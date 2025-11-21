"""
模块说明：
这是模型的基础定义。
它定义了两个类：
1. MetaModel: 一个“抽象元类”，它像一个接口，规定了所有模型“必须”实现哪些功能（函数）。
2. BaseModel: 继承了 MetaModel，并实现了所有这些功能，构建了一个通用的训练和测试框架。
这个框架的两个核心API（入口函数）是 run_train 和 run_test，它们在 main.py 中被调用。
"""

# --- 导入必要的库 ---
import torch
import numpy as np
import os.path as osp  # 用于处理文件路径
import torch.nn as nn  # PyTorch 神经网络的核心库
import torch.optim as optim  # 包含各种优化器，如 Adam, SGD
import torch.utils.data as tordata  # PyTorch 数据加载工具

from tqdm import tqdm  # 一个超好用的进度条库
from torch.cuda.amp import autocast  # 自动混合精度 (AMP)，用于加速训练
from torch.cuda.amp import GradScaler  # 配合 autocast 使用，防止梯度下溢
from abc import ABCMeta, abstractmethod  # 用于创建抽象类 (MetaModel)

# --- 从本项目中导入其他模块 ---
from . import backbones  # 导入定义好的所有骨干网络 (如 ResNet, GaitSet)
from .loss_aggregator import LossAggregator  # 损失聚合器，用于计算总损失
from data.transform import get_transform  # 数据预处理/增强函数
from data.collate_fn import CollateFn  # 如何将多张图片“打包”成一个 batch
from data.dataset import DataSet  # 数据集定义类
import data.sampler as Samplers  # 采样器，决定如何从数据集中“抽取”数据
from utils import Odict, mkdir, ddp_all_gather  # 各种工具函数
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from evaluation import evaluator as eval_functions  # 评估函数 (如计算 Rank-1 准确率)
from utils import NoOp  # 一个“什么都不做”的占位符
from utils import get_msg_mgr  # 消息/日志管理器

__all__ = ['BaseModel']  # 允许外部通过 `from . import *` 导入 BaseModel


# =================================================================================
#  1. MetaModel (抽象元类)
# =================================================================================
class MetaModel(metaclass=ABCMeta):
    """
    这是一个“抽象类”或“接口”。
    它定义了一个模型“必须”具备哪些功能，但它自己并不实现这些功能。
    任何继承了 MetaModel 的类（比如 BaseModel）都必须自己实现下面所有带 @abstractmethod 的函数。
    这是一种强制性的代码规范，确保了框架的完整性。
    """

    @abstractmethod
    def get_loader(self, data_cfg):
        """必须实现：根据数据配置(data_cfg)，获取数据加载器 (DataLoader)"""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """必须实现：根据模型配置(model_cfg)，构建你自己的网络结构"""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """必须实现：初始化你的网络参数（权重）"""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """必须实现：根据优化器配置(optimizer_cfg)，获取优化器 (Optimizer)"""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """必须实现：根据学习率策略(scheduler_cfg)，获取学习率调度器 (Scheduler)"""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """必须实现：保存检查点 (Checkpoint)，即模型权重和训练状态"""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """必须实现：从检查点 (Checkpoint) 恢复模型，用于继续训练或测试"""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """必须实现：对从 DataLoader 出来的原始输入数据进行预处理（如转为 Tensor）"""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """必须实现：执行一步训练（反向传播、梯度更新）"""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """必须实现：执行推理（在测试集上计算所有特征）"""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """必须实现：运行完整的训练流程（循环）"""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """必须实现：运行完整的测试流程"""
        raise NotImplementedError


# =================================================================================
#  2. BaseModel (核心基础模型)
# =================================================================================
class BaseModel(MetaModel, nn.Module):
    """
    核心基础模型类。
    它同时继承了 MetaModel 和 nn.Module：
    - 继承 MetaModel：意味着它“承诺”会实现 MetaModel 定义的所有抽象函数。
    - 继承 nn.Module：这是 PyTorch 中所有模型的基类。继承它，你的类才能被称为一个"PyTorch模型"，
                      才能使用 .to(device), .parameters(), .state_dict() 等核心功能。
    """

    def __init__(self, cfgs, training):
        """
        初始化函数 (Constructor)
        这是你创建 `model = BaseModel(...)` 时第一个被调用的函数。
        它的核心任务是：把一个模型跑起来所需要的所有“零件”都准备好。
        
        Args:
            cfgs (dict): 包含了所有配置信息 (data_cfg, model_cfg, trainer_cfg 等) 的总字典。
            training (bool): 当前是“训练模式” (True) 还是“测试模式” (False)。
        """

        super(BaseModel, self).__init__()  # 必须调用父类(nn.Module)的初始化

        # --- 1. 初始化基本属性 ---
        self.msg_mgr = get_msg_mgr()  # 获取日志管理器，用于打印信息
        self.cfgs = cfgs  # 把完整的配置字典存为类属性
        self.iteration = 0  # 初始化当前迭代次数为 0
        
        # 根据是训练还是测试，选择对应的配置
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("模型初始化失败：没有找到 trainer_cfg 或 evaluator_cfg")

        # --- 2. 设置混合精度 (AMP) ---
        # 如果是训练模式，并且配置中启用了 float16 (自动混合精度)
        if training and self.engine_cfg['enable_float16']:
            # GradScaler (梯度缩放器) 会在反向传播时放大loss，防止梯度小到变为0 (下溢)
            self.Scaler = GradScaler() 
        
        # --- 3. 设置保存路径 ---
        # 定义模型的输出路径，如 'output/CASIA-B/GaitSet/default_save_name'
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        # --- 4. 搭建网络并初始化 ---
        self.build_network(cfgs['model_cfg'])  # 核心：调用 build_network 搭建网络
        self.train(training)  # 核心：设置模型为 训练(True) 或 评估(False) 模式
                               # 这会影响 BatchNorm 和 Dropout 层的行为
        self.init_parameters()  # 核心：调用 init_parameters 初始化网络权重

        # --- 5. 准备数据预处理 ---
        # 获取训练时的数据预处理/增强变换 (如随机裁剪、翻转)
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])
        
        # --- 6. 准备数据加载器 (DataLoader) ---
        self.msg_mgr.log_info(cfgs['data_cfg'])  # 打印数据配置信息
        if training:
            # 如果是训练模式，创建训练数据加载器
            self.train_loader = self.get_loader(cfgs['data_cfg'], train=True)
        
        if not training or self.engine_cfg['with_test']:
            # 如果是测试模式，或者“训练时也需要测试”(with_test=True)
            # 创建测试数据加载器
            self.test_loader = self.get_loader(cfgs['data_cfg'], train=False)
            # 获取测试时的数据预处理变换 (通常只有归一化)
            self.evaluator_trfs = get_transform(cfgs['evaluator_cfg']['transform'])

        # --- 7. 设置设备 (GPU) ---
        # 在分布式训练(DDP)中，self.device 是当前进程对应的 GPU 编号 (rank)
        self.device = torch.distributed.get_rank() 
        torch.cuda.set_device(self.device)  # 告诉 PyTorch 当前进程使用哪个 GPU
        self.to(device=torch.device("cuda", self.device))  # 核心：将模型的所有参数搬到对应的 GPU 上

        # --- 8. 准备训练所需“零件” (损失、优化器、学习率) ---
        if training:
            # 创建损失聚合器，它会根据配置，管理一个或多个损失函数 (如 TripletLoss, CrossEntropyLoss)
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            # 创建优化器 (如 Adam)，它负责根据梯度更新模型参数
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            # 创建学习率调度器，它负责在训练过程中动态调整学习率 (如每10000次迭代，lr 乘以 0.1)
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        
        # --- 9. 恢复模型 (Checkpoint) ---
        restore_hint = self.engine_cfg['restore_hint']  # 获取恢复提示 (可能是迭代次数或文件路径)
        if restore_hint != 0:
            # 如果不为 0，则调用 resume_ckpt 从保存的 checkpoint 恢复模型状态
            self.resume_ckpt(restore_hint)

    # --- `__init__` 中调用的辅助函数 ---

    def get_backbone(self, backbone_cfg):
        """
        根据配置，动态地从 'backbones' 模块中获取并创建骨干网络实例。
        例如，如果 backbone_cfg['type'] 是 'GaitSet'，它会去 backbones 文件夹里
        找到 GaitSet 类，并用 backbone_cfg 里的其他参数（如 'in_channels'）来实例化它。
        """
        if is_dict(backbone_cfg):
            # 从 [backbones] 模块中，获取名字为 backbone_cfg['type'] 的那个类
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            # 获取创建这个类所需要的有效参数
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            # 实例化并返回
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            # 如果配置是一个列表（比如多分支网络），则递归地为列表中的每个配置创建 backbone
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                     for cfg in backbone_cfg])
            return Backbone
        raise ValueError("Backbone-Cfg- 必须是 dict 或 list 类型")

    def build_network(self, model_cfg):
        """
        构建网络。
        在 BaseModel 中，它只实现了最基础的：构建 `self.Backbone`。
        注意：你自己的模型 (比如 MyGaitModel) 会继承 BaseModel 并“重写”(override) 这个函数，
        在 `super().build_network(model_cfg)` 之后，添加你自己的其他层，
        比如 Head（分类头）、Reduction（池化层）等。
        """
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        """
        初始化模型所有参数（权重和偏置）。
        这是一个非常标准的权重初始化流程。
        - 遍历模型的所有模块 (self.modules())
        - 如果是卷积层 (Conv) 或线性层 (Linear)，使用 xavier_uniform_ 初始化权重，偏置设为 0。
        - 如果是批归一化层 (BatchNorm)，权重设为 1，偏置设为 0。
        （目的是让 BatchNorm 在训练刚开始时接近于一个“恒等”变换）
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:  # affine=True 意味着 BN 层有可学习的 γ 和 β 参数
                    nn.init.normal_(m.weight.data, 1.0, 0.02)  # 权重(γ)初始化为均值1、方差0.02的正态分布
                    nn.init.constant_(m.bias.data, 0.0)  # 偏置(β)初始化为 0

    def get_loader(self, data_cfg, train=True):
        """
        创建并返回一个 PyTorch DataLoader。
        
        Args:
            data_cfg: 数据配置
            train (bool): 是为训练集还是测试集创建
        
        Returns:
            tordata.DataLoader: PyTorch 数据加载器实例
        """
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        
        # 1. 创建 DataSet 实例
        #    DataSet 知道如何根据索引 (index) 从磁盘读取一条数据 (如一张图片和它的标签)
        dataset = DataSet(data_cfg, train)

        # 2. 创建 Sampler 实例
        #    Sampler 决定了如何“采样”数据。它不是一次取一个，而是决定“一批(batch)”数据
        #    应该包含哪些索引。这在步态识别中至关重要，比如 TripletSampler 
        #    会确保一个 batch 中包含 P 个人的 K 张图片。
        Sampler = get_attr_from([Samplers], sampler_cfg['type'])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=['sample_type', 'type'])
        sampler = Sampler(dataset, **vaild_args)

        # 3. 创建 DataLoader 实例
        loader = tordata.DataLoader(
            dataset=dataset,  # 数据集
            batch_sampler=sampler,  # 批采样器 (注意：用了 batch_sampler, 就不能再用 batch_size, shuffle, drop_last)
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),  # 整理函数：告诉 DataLoader 如何将采样器给的一组数据
                                                                 # (如 [img1, lbl1], [img2, lbl2]) 打包成一个批张量 (batch_tensor)
            num_workers=data_cfg['num_workers']  # 使用多少个子进程在后台加载数据，加快速度
        )
        return loader

    def get_optimizer(self, optimizer_cfg):
        """
        创建并返回一个优化器 (Optimizer)。
        """
        self.msg_mgr.log_info(optimizer_cfg)  # 打印优化器配置
        # 动态获取优化器类，如 torch.optim.Adam
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        # 获取有效参数，如 'lr', 'weight_decay'
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        
        # 实例化优化器
        optimizer = optimizer(
            # 核心：只将需要计算梯度(requires_grad=True)的参数交给优化器
            # 这允许你“冻结”某些层（比如预训练的 backbone）
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        """
        创建并返回一个学习率调度器 (Scheduler)。
        """
        self.msg_mgr.log_info(scheduler_cfg)  # 打印调度器配置
        # 动态获取调度器类，如 torch.optim.lr_scheduler.StepLR
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        # 获取有效参数，如 'step_size', 'gamma'
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        # 实例化调度器，并将其与优化器绑定
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    # --- Checkpoint (模型保存与加载) 相关函数 ---

    def save_ckpt(self, iteration):
        """
        保存模型检查点 (Checkpoint)。
        """
        # 只在主进程 (rank=0) 上执行保存操作，防止多 GPU 同时写入造成冲突
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))  # 创建保存目录
            save_name = self.engine_cfg['save_name']
            
            # 创建一个字典，包含所有需要保存的状态
            checkpoint = {
                'model': self.state_dict(),  # 核心：模型的权重
                'optimizer': self.optimizer.state_dict(),  # 优化器的状态 (如 Adam 的动量)
                'scheduler': self.scheduler.state_dict(),  # 调度器的状态 (如当前步数)
                'iteration': iteration  # 当前的迭代次数
            }
            # 保存到 .pt 文件
            torch.save(checkpoint,
                       osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    def _load_ckpt(self, save_name):
        """
        实际执行加载 checkpoint 文件的内部函数。
        """
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']  # 是否“严格”加载

        # 加载 checkpoint 文件，并将其映射到当前 GPU
        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']  # 取出模型权重

        if not load_ckpt_strict:
            # 如果不是严格模式，打印出哪些权重被成功加载了
            # 这在“迁移学习”时很有用，比如你只加载了 backbone 的权重
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        # 核心：将加载的权重载入到当前模型中
        # strict=True: 要求 checkpoint 的键值与当前模型严格一致
        # strict=False: 允许不一致，只加载键值匹配的权重
        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)

        # 如果是训练模式，还需要恢复优化器和调度器的状态
        if self.training:
            # 如果不重置优化器，并且 checkpoint 中有，就加载
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(f"警告：未从 {save_name} 恢复优化器状态！")
            
            # 如果不重置调度器，并且 checkpoint 中有，就加载
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(f"警告：未从 {save_name} 恢复调度器状态！")
        self.msg_mgr.log_info(f"成功从 {save_name} 恢复模型参数！")

    def resume_ckpt(self, restore_hint):
        """
        恢复模型的入口函数。它会解析 `restore_hint` 并调用 `_load_ckpt`。
        """
        if isinstance(restore_hint, int):
            # 如果 restore_hint 是一个数字 (如 80000)，说明是迭代次数
            save_name = self.engine_cfg['save_name']
            # 自动拼出 checkpoint 文件的完整路径
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint  # 同步迭代次数
        elif isinstance(restore_hint, str):
            # 如果是一个字符串，说明是 checkpoint 文件的完整路径
            save_name = restore_hint
            self.iteration = 0  # 迭代次数从 0 开始
        else:
            raise ValueError("restore_hint 必须是 int 或 string 类型")
        
        self._load_ckpt(save_name)

    # --- 训练/测试 辅助函数 ---

    def fix_BN(self):
        """
        固定 BatchNorm 层的统计量 (均值和方差)。
        在微调 (Fine-tuning) 时常用。
        它通过将所有 BatchNorm 层设置为 .eval() 模式来实现。
        这样，BN 层会使用训练时学到的全局均值/方差，而不是当前 batch 的。
        """
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def inputs_pretreament(self, inputs):
        """
        对 DataLoader 出来的“一个批次”数据进行最终的预处理。
        
        Args:
            inputs: 从 DataLoader 出来的原始批次数据 (通常是 list of numpy arrays)
        
        Returns:
            一个元组 (tuple)，包含准备好喂给模型的张量 (Tensors)
        """
        # 1. 解包
        #    seqs_batch: 序列数据 (如剪影)
        #    labs_batch: 标签 (ID)
        #    typs_batch: 类型 (如 'nm', 'cl', 'bg')
        #    vies_batch: 视角 (如 '090')
        #    seqL_batch: 序列长度 (如果序列不等长)
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        
        # 2. 选择对应的变换
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError("输入数据流的数量和 transform 的数量不一致！")
            
        requires_grad = bool(self.training)  # 训练时需要梯度，测试时不需要

        # 3. 核心：数据变换与格式转换
        #    这是一个两层循环：
        #    - 外层循环 `zip(seq_trfs, seqs_batch)`: 遍历每一种数据流 (如 $B_1^{ap}, B_1^{de}$)
        #    - 内层循环 `[trf(fra) for fra in seq]`: 对这个数据流中的每个样本 (seq)，
        #      再对样本中的每一帧 (fra)，应用预处理函数 (trf)，比如归一化。
        #    - `np.asarray(...)`: 将处理后的帧重新组合成一个 numpy 数组。
        #    - `np2var(...)`: 将 numpy 数组转换为 PyTorch Tensor，并搬到 GPU 上。
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]

        # 4. 转换其他元数据
        typs = typs_batch
        vies = vies_batch
        labs = list2var(labs_batch).long()  # 标签转为 Tensor
        
        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()  # 序列长度转为 Tensor
        seqL = seqL_batch

        # 5. 处理不等长序列 (如果需要)
        #    这是为了“打包序列”(Packed Sequence) 准备的，一种处理RNN不等长输入的技巧
        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
            
        del seqs  # 及时释放内存
        return ipts, labs, typs, vies, seqL

    def train_step(self, loss_sum) -> bool:
        """
        执行“一步”训练，即：反向传播 + 梯度更新 + 学习率调整。
        
        Args:
            loss_sum: 当前 batch 的总损失 (一个标量 Tensor)
        
        Returns:
            bool: 训练是否成功 (如果不成功，比如梯度爆炸，可以跳过此步)
        """

        # 1. 梯度清零
        #    如果不清零，梯度会累积
        self.optimizer.zero_grad()
        
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning("警告：损失值 <= 1e-9，但训练仍将继续！")

        # 2. 反向传播 (计算梯度)
        if self.engine_cfg['enable_float16']:  # --- 2a. 混合精度 (AMP) 模式 ---
            # self.Scaler.scale(loss_sum): 
            #   在反向传播前，将 loss 乘以一个很大的“缩放因子” (如 65536)
            #   这样可以防止 float16 格式下很小的梯度“下溢”为 0
            self.Scaler.scale(loss_sum).backward()
            
            # self.Scaler.step(self.optimizer):
            #   1. 将参数的梯度“反缩放”回原始大小
            #   2. 检查梯度中是否有 NaN 或 Inf (梯度爆炸)
            #   3. 如果没有，则调用 self.optimizer.step() 更新参数
            #   4. 如果有，则跳过此次更新
            self.Scaler.step(self.optimizer)
            
            # self.Scaler.update():
            #   动态调整“缩放因子”。如果连续多次没有 NaN，就尝试增大它；
            #   如果出现了 NaN，就大幅减小它。
            scale = self.Scaler.get_scale()
            self.Scaler.update()

            # 检查是否因为梯度爆炸 (NaN/Inf) 而跳过了更新
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug(f"训练步骤跳过。缩放因子从 {scale} 变为 {self.Scaler.get_scale()}")
                return False  # 返回 False，主循环会 continue
        else:  # --- 2b. 标准 (float32) 模式 ---
            loss_sum.backward()  # 计算梯度
            self.optimizer.step()  # 更新参数

        # 3. 更新迭代次数和学习率
        self.iteration += 1  # 迭代计数器 +1
        self.scheduler.step()  # 学习率调度器走一步
        return True  # 返回 True，表示此步训练成功

    def inference(self, rank):
        """
        在测试集上执行“推理”，即计算所有数据的特征向量 (Feature)。
        
        Args:
            rank: 当前 GPU 的编号
            
        Returns:
            Odict: 一个字典，包含了所有样本的特征、标签、类型、视角
        """
        total_size = len(self.test_loader)  # 测试集总样本数
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')  # 在主进程上显示进度条
        else:
            pbar = NoOp()  # 其他进程不显示
            
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()  # 用 Odict 来收集所有批次的结果

        # 遍历测试集
        for inputs in self.test_loader:
            # 1. 预处理
            ipts = self.inputs_pretreament(inputs)
            
            # 2. 推理（前向传播）
            with autocast(enabled=self.engine_cfg['enable_float16']):
                # 核心：调用模型的 forward 函数（在你自己的模型中定义）
                # `retval` 是你 forward 函数的返回值
                retval = self.forward(ipts)
                
            # 3. 收集结果
            inference_feat = retval['inference_feat']  # 拿到推理特征
            
            # 4. DDP 同步
            #    在 DDP 测试中，每个 GPU 只处理了一部分数据。
            #    `ddp_all_gather` 会从所有 GPU 收集 `inference_feat`，
            #    并把完整的结果广播回每个 GPU。
            for k, v in inference_feat.items():
                inference_feat[k] = ddp_all_gather(v, requires_grad=False)
            del retval
            
            # 5. 格式转换 (Tensor -> Numpy)
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)  # ts2np = tensor to numpy
            
            info_dict.append(inference_feat)  # 将这个 batch 的结果存入 Odict
            
            # --- 更新进度条 ---
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
            
        pbar.close()
        
        # 6. 最终整理
        #    此时 `info_dict` 里的每个 value 都是一个 list of numpy arrays (每个array来自一个batch)
        #    `np.concatenate` 将它们合并成一个大的 numpy array
        #    `[:total_size]` 是为了去除 DDP 采样时可能重复添加的 padding 数据
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    # =================================================================================
    #  3. 静态方法 (Static Methods) - 框架的入口
    # =================================================================================
    # @staticmethod 意味着这个函数不依赖于 `self` (类的实例)。
    # 你可以直接通过 `BaseModel.run_train(model_instance)` 来调用它。
    # 这是一种将“执行逻辑”与“模型定义”解耦的漂亮写法。

    @staticmethod
    def run_train(model):
        """
        完整的“训练循环” (Train Loop)。
        
        Args:
            model: 已经初始化好的 BaseModel 实例。
        """
        # `model.train_loader` 是一个可迭代对象，会不断产生数据
        for inputs in model.train_loader:
            # 1. 数据预处理
            ipts = model.inputs_pretreament(inputs)
            
            # 2. 前向传播
            #    使用 autocast 开启混合精度
            with autocast(enabled=model.engine_cfg['enable_float16']):
                # 核心：调用 model.forward(ipts)
                retval = model(ipts)  
                # `retval` 是你 forward 函数的返回值，必须包含 'training_feat'
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval

            # 3. 计算损失
            #    将模型输出的特征和标签送入损失聚合器
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            
            # 4. 反向传播与优化
            #    执行梯度计算、梯度更新、学习率调整
            ok = model.train_step(loss_sum)
            if not ok:  # 如果 train_step 返回 False (如梯度爆炸)，则跳过此批次
                continue

            # 5. 日志与可视化
            visual_summary.update(loss_info)  # 把损失信息也加入到可视化摘要
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']  # 记录当前学习率
            model.msg_mgr.train_step(loss_info, visual_summary)  # 打印日志、写入 TensorBoard

            # 6. 保存与测试
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # --- 保存模型 ---
                model.save_ckpt(model.iteration)

                # --- 运行测试 ---
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()  # 关键：切换到评估模式
                    result_dict = BaseModel.run_test(model)  # 调用测试流程
                    model.train()  # 关键：切换回训练模式
                    
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()  # 如果需要，重新固定 BN
                        
                    if result_dict:
                        # 将测试结果 (如 Rank-1) 写入 TensorBoard
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            
            # 7. 检查是否结束
            if model.iteration >= model.engine_cfg['total_iter']:
                break  # 达到总迭代次数，跳出训练循环

    @staticmethod
    def run_test(model):
        """
        完整的“测试流程” (Test Loop)。
        
        Args:
            model: 已经初始化好的 BaseModel 实例。
        """
        evaluator_cfg = model.cfgs['evaluator_cfg']
        
        # 在 DDP 测试中，为保证数据同步，batch_size 必须等于 GPU 数量
        if torch.distributed.get_world_size() != evaluator_cfg['sampler']['batch_size']:
            raise ValueError("测试模式下，batch size 必须等于 GPU 数量！")
            
        rank = torch.distributed.get_rank()  # 获取当前 GPU 编号
        
        # 1. 推理
        #    `torch.no_grad()`: 关键！
        #    关闭梯度计算，能极大节省显存并加快速度。
        with torch.no_grad():
            info_dict = model.inference(rank)  # 调用 inference 获取所有特征
            
        # 2. 评估
        #    只有主进程 (rank=0) 才执行评估和打印
        if rank == 0:
            loader = model.test_loader
            # 从数据集中获取真实的标签、类型、视角列表
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list

            # 将这些真实信息加入到 info_dict 中
            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            # 3. 调用评估函数
            eval_func_name = evaluator_cfg.get('eval_func', 'identification')
            
            # 动态获取评估函数 (如 'identification')
            eval_func = getattr(eval_functions, eval_func_name)
            
            # 获取评估函数需要的参数 (如 'metric')
            valid_args = get_valid_args(
                eval_func, evaluator_cfg, ['metric'])
            
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            
            # 最终：调用评估函数，传入包含(特征、标签、视角)的字典，返回评估结果
            return eval_func(info_dict, dataset_name, **valid_args)