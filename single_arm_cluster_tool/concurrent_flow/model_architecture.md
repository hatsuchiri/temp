# 模型架构详解

本文档详细描述了 `CONCATNet` 模型的架构，包括从输入到输出的完整流程、各层的输入输出维度以及相关参数。

## 1. 模型整体结构

`CONCATNet` 是一个用于集群工具调度的强化学习模型，主要由两个核心组件组成：

- **Encoder**：负责将环境状态编码为低维嵌入向量
- **Decoder**：负责基于编码后的嵌入向量生成动作概率分布

## 2. 输入与输出

### 2.1 输入
- **环境状态** (`State` 对象)：包含以下关键信息
  - `clock`：当前时间
  - `loc_process_end_time`：各处理模块的处理结束时间
  - `loadlock1_wafer_recipe`、`loadlock2_wafer_recipe`：加载锁中的晶圆配方
  - `loc_hold_wafer`：各处理模块当前持有的晶圆
  - `loc_stage`：各处理模块当前的处理阶段
  - `robot_loc`：机器人当前位置
  - `action_mask`：可用动作的掩码

### 2.2 输出
- **动作** (`batch_action`)：选择的动作索引，形状为 `(batch_size,)`
- **动作概率** (`batch_prob`)：选择动作的概率，形状为 `(batch_size,)`

## 3. Encoder 架构

### 3.1 Encoder 整体结构

```
Encoder
├── init_embedding (LotStageEmbedding)
└── layers (nn.ModuleList)
    ├── EncoderLayer 0
    │   ├── row_encoding_block (EncodingBlock)
    │   └── col_encoding_block (EncodingBlock)
    ├── EncoderLayer 1
    │   ├── row_encoding_block (EncodingBlock)
    │   └── col_encoding_block (EncodingBlock)
    └── EncoderLayer 2
        ├── row_encoding_block (EncodingBlock)
        └── col_encoding_block (EncodingBlock)
```

### 3.2 输入输出维度

| 模块 | 输入 | 输出 |
|------|------|------|
| **LotStageEmbedding** | 环境状态 | row_embed: (batch_size, num_lot_types, embedding_dim)<br>col_embed: (batch_size, num_stages, embedding_dim)<br>cost_mat: (batch_size, num_lot_types, num_stages) |
| **EncoderLayer** | row_embed: (batch_size, num_lot_types, embedding_dim)<br>col_embed: (batch_size, num_stages, embedding_dim)<br>cost_mat: (batch_size, num_lot_types, num_stages) | row_embed_out: (batch_size, num_lot_types, embedding_dim)<br>col_embed_out: (batch_size, num_stages, embedding_dim) |
| **EncodingBlock** | input1: (batch_size, N, embedding_dim)<br>input2: (batch_size, M, embedding_dim)<br>cost_mat: (batch_size, N, M) | output: (batch_size, N, embedding_dim) |

### 3.3 EncodingBlock 内部结构

```
EncodingBlock
├── Wq (nn.Linear): embedding_dim → head_num * qkv_dim
├── Wk (nn.Linear): embedding_dim → head_num * qkv_dim
├── Wv (nn.Linear): embedding_dim → head_num * qkv_dim
├── mixed_score_MHA (MixedScore_MultiHeadAttention)
├── multi_head_combine (nn.Linear): head_num * qkv_dim → embedding_dim
├── add_n_normalization_1 (AddAndInstanceNormalization)
├── feed_forward (FeedForward)
└── add_n_normalization_2 (AddAndInstanceNormalization)
```

## 4. Decoder 架构

### 4.1 Decoder 整体结构

```
Decoder
├── pm_dynamic_linear (nn.Linear): 2 → embedding_dim
├── pm_concat_liear (nn.Linear): 3*embedding_dim → embedding_dim
├── action_time_linear (nn.Linear): 1 → embedding_dim
├── loadport_context
├── pm_context
├── robot_context
├── concat_context (nn.Sequential)
│   ├── nn.Linear: 3*embedding_dim → embedding_dim
│   ├── nn.ReLU()
│   └── nn.Linear: embedding_dim → embedding_dim
├── Wq_1, Wq_2, Wq_3 (nn.Linear): embedding_dim → head_num * qkv_dim
├── Wk (nn.Linear): 3*embedding_dim → head_num * qkv_dim
├── Wv (nn.Linear): 3*embedding_dim → head_num * qkv_dim
├── Wshk (nn.Linear): 3*embedding_dim → embedding_dim
├── multi_head_combine (nn.Linear): head_num * qkv_dim → embedding_dim
└── softmax (F.softmax): 生成动作概率
```

### 4.2 输入输出维度

| 模块 | 输入 | 输出 |
|------|------|------|
| **Decoder** | encoded_row: (batch_size, num_lot_types, embedding_dim)<br>encoded_col: (batch_size, num_stages, embedding_dim)<br>state: 环境状态 | probs: (batch_size, action_space_size) |
| **get_PM_embedding** | encoded_row, encoded_col, state | pm_embeddings: (batch_size, num_pm, embedding_dim)<br>pm_wafer_embeddings: (batch_size, num_pm, embedding_dim) |
| **get_action_embedding** | encoded_row, encoded_col, state, pm_embeddings, pm_wafer_embeddings | action_embedding: (batch_size, num_actions, 3*embedding_dim) |
| **concat_context** | ll_context: (batch_size, embedding_dim)<br>pm_context: (batch_size, embedding_dim)<br>robot_context: (batch_size, embedding_dim) | context: (batch_size, 1, embedding_dim) |

## 5. 关键参数

| 参数名称 | 默认值 | 描述 | 位置 |
|----------|--------|------|------|
| `embedding_dim` | 256 | 嵌入向量维度 | model_params |
| `sqrt_embedding_dim` | 16 | 嵌入向量维度的平方根 | model_params |
| `head_num` | 16 | 多头注意力的头数 | model_params |
| `qkv_dim` | 16 | 查询、键、值向量的维度 | model_params |
| `sqrt_qkv_dim` | 4 | qkv_dim的平方根 | model_params |
| `encoder_layer_num` | 3 | Encoder的层数 | model_params |
| `logit_clipping` | 10 | 对数概率裁剪值 | model_params |
| `ff_hidden_dim` | 512 | 前馈网络的隐藏层维度 | model_params |
| `ms_hidden_dim` | 16 | 混合分数网络的隐藏层维度 | model_params |
| `ms_layer1_init` | (1/2)**(1/2) | 混合分数网络第一层的初始化缩放因子 | model_params |
| `ms_layer2_init` | (1/16)**(1/2) | 混合分数网络第二层的初始化缩放因子 | model_params |
| `eval_type` | 'softmax' | 评估时的动作选择类型 | model_params |
| `normalize` | 'instance' | 归一化类型 | model_params |

## 6. 前向传播流程

### 6.1 编码阶段 (Encoding)

1. **初始化嵌入**：`LotStageEmbedding` 将环境状态转换为初始嵌入向量
2. **多层编码**：通过3个 `EncoderLayer` 对嵌入向量进行编码
   - 每个 `EncoderLayer` 包含两个 `EncodingBlock`，分别处理行和列嵌入
   - 每个 `EncodingBlock` 使用多头注意力机制和前馈网络
3. **输出编码结果**：得到最终的 `row_embed` 和 `col_embed`

### 6.2 解码阶段 (Decoding)

1. **生成PM嵌入**：基于编码后的嵌入向量和环境状态生成处理模块的嵌入
2. **生成动作嵌入**：结合PM嵌入、动作持续时间等信息生成动作嵌入
3. **生成上下文嵌入**：结合加载锁上下文、PM上下文和机器人上下文
4. **多头注意力**：使用上下文嵌入作为查询，动作嵌入作为键和值
5. **生成动作概率**：通过softmax生成最终的动作概率分布

### 6.3 动作选择

- **训练模式**：使用 multinomial 从概率分布中采样动作
- **评估模式**：选择概率最高的动作

## 7. 梯度流动

### 7.1 梯度计算路径

```
动作概率 → 对数概率 → 损失计算 → 反向传播 → 解码器参数 → 编码器参数
```

### 7.2 关键注意点

- **编码嵌入** (`row_embed` 和 `col_embed`) 是需要梯度的，它们的梯度会反向传播到编码器参数
- **动作概率** (`batch_prob`) 是解码器的输出，其梯度会反向传播到解码器参数
- **损失函数**：使用优势函数加权的策略梯度损失

## 8. 模型参数统计

| 组件 | 参数数量 | 主要参数 |
|------|----------|----------|
| **Encoder** | 约 1.5M | 各层的 Wq, Wk, Wv, multi_head_combine, feed_forward |
| **Decoder** | 约 2.0M | pm_dynamic_linear, pm_concat_liear, Wq_1/2/3, Wk, Wv, concat_context |
| **总计** | 约 3.5M | 所有可学习参数 |

## 9. 输入输出示例

### 9.1 输入示例

```python
# 环境状态示例
batch_state = State(
    clock=torch.tensor([[0.0], [0.0]]),  # (batch_size, 1)
    loc_process_end_time=torch.zeros((2, 6)),  # (batch_size, num_pm)
    loadlock1_wafer_recipe=torch.tensor([0, 1]),  # (batch_size,)
    loadlock2_wafer_recipe=torch.tensor([1, 0]),  # (batch_size,)
    loc_hold_wafer=torch.tensor([[-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]]),  # (batch_size, num_pm)
    loc_stage=torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),  # (batch_size, num_pm)
    robot_loc=torch.tensor([0, 0]),  # (batch_size,)
    action_mask=torch.ones((2, 8))  # (batch_size, num_actions)
)
```

### 9.2 输出示例

```python
# 模型输出
batch_action, batch_prob = model(batch_state)
# batch_action: tensor([3, 5])  # (batch_size,)
# batch_prob: tensor([0.125, 0.125])  # (batch_size,)
```

## 10. 代码优化建议

1. **参数初始化**：考虑为第一个编码层使用不同的参数初始化策略，以避免梯度消失
2. **梯度检查**：在训练过程中添加梯度检查，及时发现梯度异常
3. **模型压缩**：考虑使用知识蒸馏或其他模型压缩技术减少参数量
4. **并行计算**：利用PyTorch的并行计算能力加速模型训练

## 11. 总结

`CONCATNet` 是一个结构复杂但设计合理的强化学习模型，通过编码器-解码器架构有效地处理集群工具调度问题。模型的关键优势在于：

- **强大的状态表示能力**：通过多层编码捕获环境状态的复杂特征
- **灵活的动作生成**：基于上下文信息生成合理的动作概率分布
- **端到端训练**：支持从环境交互直接学习最优策略

通过本文档的详细描述，希望能够帮助理解模型的内部工作机制，为后续的模型改进和应用提供参考。
