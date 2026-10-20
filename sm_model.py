import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.autograd import Function
from sklearn.utils.class_weight import compute_class_weight

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x,lambda_) :
        ctx.lambda_=lambda_
        return x.clone()
    @staticmethod
    def backward(ctx,grads):
        lambda_=ctx.lambda_
        lambda_=grads.new_tensor(lambda_)
        dx=-lambda_*grads
        return dx, None
class GradientReversal(torch.nn.Module):
    def __init__(self,lambda_=0.5):
        super(GradientReversal, self).__init__()
        self.lambda_=lambda_
    def forward(self,x):
        return GradientReversalFunction.apply(x,self.lambda_)





def dbma_attention(q, k, v, v_mask=None, dropout=None,
                   block_size=64, num_global=4,
                   expansion_ratio=0.5):
    """
    Dynamic Block-Mixed Attention (DBMA)
    
    """
   
    if q.dim() == 2:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    elif q.dim() != 3:
        raise ValueError(f"输入维度错误，期望3D张量，实际得到{q.shape}")

    batch, seq_len, dim = q.shape
    device = q.device

    # ===== 1. 动态分块策略重构 =====
    def calc_adaptive_block(seq_len):
        if seq_len <= 1:
            return seq_len

        min_block = max(1, block_size // 8)
        max_block = min(1024, block_size * 2)

        base = max(1, seq_len // 16)
        log_value = torch.log2(torch.tensor(base)).item()
        adaptive_block = 2  ** max(2, int(log_value))
        return min(max_block, max(min_block, adaptive_block))

    adaptive_block = calc_adaptive_block(seq_len)

    # ===== 2. 分块维度动态对齐 =====
    if adaptive_block == 0:
        adaptive_block = 1

    pad_len = (adaptive_block - (seq_len % adaptive_block)) % adaptive_block
    padded_length = seq_len + pad_len

    num_blocks = max(1, (padded_length + adaptive_block - 1) // adaptive_block)
    max_retry = 3
    for _ in range(max_retry):
        if padded_length % num_blocks == 0:
            break
        num_blocks = min(max(1, padded_length // max(adaptive_block, 1)), 16)
        adaptive_block = (padded_length + num_blocks - 1) // num_blocks
        pad_len = adaptive_block * num_blocks - seq_len
        padded_length = seq_len + pad_len

    if padded_length % num_blocks != 0:
        num_blocks = max(1, padded_length)
        adaptive_block = padded_length // num_blocks

    # ===== 3. 张量填充验证 =====
    q_padded = F.pad(q, (0, 0, 0, pad_len))
    k_padded = F.pad(k, (0, 0, 0, pad_len))
    v_padded = F.pad(v, (0, 0, 0, pad_len))

    # ===== 4. 分块变形保护 =====
    actual_length = q_padded.shape[1]
    adaptive_block = actual_length // num_blocks

    for _ in range(3):
        if actual_length == num_blocks * adaptive_block:
            break
        adaptive_block = (actual_length + num_blocks - 1) // num_blocks
        num_blocks = actual_length // adaptive_block
        pad_len = adaptive_block * num_blocks - seq_len

    try:
        q_blocks = q_padded.view(batch, num_blocks, adaptive_block, dim)
        k_blocks = k_padded.view(batch, num_blocks, adaptive_block, dim)
        v_blocks = v_padded.view(batch, num_blocks, adaptive_block, dim)
    except RuntimeError as e:
        # 关键修正点：统一维度处理 [6,7](@ref)
        q_flat = q_padded.view(batch, -1, dim)
        k_flat = k_padded.view(batch, -1, dim)
        v_flat = v_padded.view(batch, -1, dim)  # 新增v的维度对齐

        global_attn = torch.einsum('bnd,bmd->bnm', q_flat, k_flat) / (dim  ** 0.5)
        return torch.matmul(global_attn, v_flat)[:, :seq_len, :]  # 截断到原始长度

    # ===== 5. 混合注意力计算优化 =====
    if adaptive_block >= 1:
        local_q = q_blocks.reshape(batch * num_blocks, adaptive_block, 1, dim)
        local_k = k_blocks.reshape(batch * num_blocks, adaptive_block, 1, dim)
        local_attn = torch.einsum('bnqd,bmkd->bnqk', local_q, local_k) / (dim  ** 0.5)
    else:
        q_flat = q.view(batch, -1, dim)
        k_flat = k.view(batch, -1, dim)
        local_attn = torch.einsum('bnd,bmd->bnm', q_flat, k_flat) / (dim  ** 0.5)
        local_attn = local_attn.unsqueeze(1)

    # ===== 6. 全局路由生成 =====
    compressed_dim = max(1, int(dim * expansion_ratio))
    global_gate = nn.Sequential(
        nn.Linear(dim, compressed_dim),
        nn.GELU(),
        nn.Linear(compressed_dim, num_global),
        nn.Softmax(-1)
    ).to(device)

    # ===== 7. 动态路由计算优化 ===== [8](@ref)
    block_centroids = q_blocks.mean(2)
    route_weights = global_gate(block_centroids)

    global_attn = torch.matmul(
        route_weights.clamp(min=1e-6),
        k_blocks.reshape(batch, num_blocks, -1)
    ).view(batch, num_global, adaptive_block, dim)

    # ===== 8. 分层融合与输出 =====
    fused_attn = torch.cat([
        local_attn.view(batch, -1, dim),
        torch.einsum('bnd,bkfd->bnfd', route_weights, global_attn).reshape(batch, -1, dim)
    ], dim=1)[:, :seq_len, :]

    # 最终投影
    fusion_proj = nn.Linear(fused_attn.shape[-1], dim).to(device)
    final_output = fusion_proj(fused_attn)

    if v_mask is not None:
        if v_mask.dim() == 1:
            v_mask = v_mask.unsqueeze(0)
        extended_mask = (1.0 - v_mask[:, :seq_len].unsqueeze(-1)) * -1e9
        final_output += extended_mask.to(device)

    if dropout is not None:
        final_output = dropout(final_output)

        # ===== 维度对齐修正 ===== [6,8](@ref)
        # 确保输出维度为 [batch, seq_len, dim]
    if final_output.dim() == 4:  # 处理可能的四维输出
        final_output = final_output.squeeze(1)
    return final_output[:, :, :]


def emb_proj(fp, fc):
    """支持三维输入的改进版投影函数"""
    # 保存原始维度用于最终恢复
    original_fp_dim = fp.dim()
    original_fc_dim = fc.dim()

    # ===== 维度修正 =====
    if fp.dim() == 2:
        fp = fp.unsqueeze(0)  # [1, seq_len, dim]
    if fc.dim() == 2:
        fc = fc.unsqueeze(0)  # [1, seq_len, dim]

    batch_size, seq_len, dim = fp.shape

    # ===== 批量矩阵乘法优化 =====
    fc_t = fc.permute(0, 2, 1)  # [batch, dim, seq_len]
    x_y = torch.bmm(fp, fc_t)  # [batch, seq_len, seq_len]
    x_y_diag = torch.diagonal(x_y, dim1=1, dim2=2)  # [batch, seq_len]

    # ===== 投影计算优化 =====
    x_y_expanded = x_y_diag.unsqueeze(-1)  # [batch, seq_len, 1]
    x_y_y = x_y_expanded * fc  # [batch, seq_len, dim]
    y_length = torch.sum(fc  ** 2, dim = -1, keepdim = True) + 1e-8  # [batch, seq_len, 1]
    proj = x_y_y / y_length  # [batch, seq_len, dim]

    # ===== 维度对齐修正 =====
    # 关键修改点：根据原始输入维度恢复形状
    if original_fp_dim == 2 and original_fc_dim == 2:
        proj = proj.squeeze(0)  # 恢复二维：[seq_len, dim]
    elif original_fp_dim == 3 and seq_len == 1:
        proj = proj.squeeze(1)  # 处理序列长度为1的特殊情况

    return proj



class Model(nn.Module):
    def __init__(self, model_name, labels, loss_type=1, checkpoint=None, debug=False):
        super(Model, self).__init__()
        self.label_num = labels
        self.debug = debug
        self.config = AutoConfig.from_pretrained(model_name)
        self.code=nn.Embedding(1,self.config.hidden_size)
        self.gradient_reversal=GradientReversal()
        self.loss_type = loss_type

        if "deberta" in model_name.lower() or "funnel" in model_name.lower():
            self.pooler = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Tanh()
            )
        self.classifier = nn.Linear(self.config.hidden_size, labels)
        self.kw_con_classifier = nn.Linear(self.config.hidden_size, 1)
        self.attention_classifier=nn.Sequential(GradientReversal(),
                                  nn.Linear(self.config.hidden_size,self.config.hidden_size),
                                  nn.Tanh(),
                                  nn.Linear(self.config.hidden_size,self.label_num)
        )
        self.rev_loss=nn.CrossEntropyLoss()
        if "funnel" in model_name.lower():
            self.dropout = nn.Dropout(self.config.hidden_dropout)
        else:
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.apply(self._init_weights)

        self.encoder = AutoModel.from_pretrained(model_name)

        if checkpoint is not None:
            cp = torch.load(checkpoint, map_location=lambda storage, loc: storage)
            self.load_state_dict(cp, strict=True)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def forward(self, input_ids, token_type_ids, attention_mask, labels,
                keyword_mask, context_mask, special_mask):


        if not self.training and not self.debug:
            output_all = self.encoder(input_ids, attention_mask, token_type_ids, return_dict=True)
            
            if "pooler_output" in output_all.keys():
                
                sequence_out = output_all.last_hidden_state
                
                common_emb=dbma_attention(self.code(torch.zeros(1, dtype=torch.long).cuda()), sequence_out.detach(),
                                           sequence_out.detach(), attention_mask, dropout=None,
                               block_size=64, num_global=4,
                               expansion_ratio=0.5)
                
                sentence_proj = emb_proj(output_all.pooler_output, common_emb)
                sentence_embeddings = emb_proj(output_all.pooler_output, output_all.pooler_output- sentence_proj)
                
                logits_all = self.classifier(self.dropout(output_all.pooler_output))
                logits_kw = self.classifier(self.dropout( sentence_embeddings))
                
                logits_detail = self.classifier(self.dropout(sentence_proj))
                
            else:
                pooler_out_all = self.pooler(output_all.last_hidden_state[:, 0])
                
                sequence_out = output_all.last_hidden_state
                
                common_emb = dbma_attention(self.code(torch.zeros(1, dtype=torch.long).cuda()), sequence_out.detach(),
                                            sequence_out.detach(), attention_mask, dropout=None,
                                            block_size=64, num_global=4,
                                            expansion_ratio=0.5)
                sentence_proj = emb_proj(pooler_out_all, common_emb)
                sentence_embeddings = emb_proj(pooler_out_all, pooler_out_all - sentence_proj)
                logits_kw = self.classifier(self.dropout(sentence_embeddings))
                
                logits_all = self.classifier(self.dropout(pooler_out_all))
                
                logits_detail = self.classifier(self.dropout(sentence_proj))
                
            all_labels = set(labels.cpu())
      
            labels_np = np.array(list(all_labels))
            final_labels = np.unique(labels_np)
            labels_weight_np = compute_class_weight(class_weight="balanced", classes=final_labels, y=labels_np)
            labels_weight_list = labels_weight_np.tolist()
            final_labels_list = final_labels.tolist()
            #label_fact = [0, 1, 2,3]
            #label_fact = [0, 1]
            label_fact = [0, 1,2]
            labels_weight_modern = [1e-5 for element in label_fact if element not in final_labels_list]
            labels_weight_list += labels_weight_modern
            labels_weight_list_np = np.array(labels_weight_list)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            labels_weight_balance = torch.tensor(labels_weight_list_np).to(device).float()
            labels_weight_balance = labels_weight_balance.float()
            rev_logit = self.attention_classifier(common_emb)
            
            if rev_logit.dim() == 3:
                # 处理三维输入 [batch, seq_len, label_num]
                if rev_logit.size(1) != 1:  # 常规序列标注
                    rev_logit = rev_logit.view(-1, self.label_num)
                else:  # 处理单元素序列的特殊情况
                    rev_logit = rev_logit.squeeze(1)  # 降维到 [batch, label_num]

            elif rev_logit.dim() == 2:
                # 处理二维输入 [batch, label_num]
                if rev_logit.size(0) != labels.size(0):
                    rev_logit = rev_logit.repeat_interleave(labels.size(0) // rev_logit.size(0), dim=0)
            else:
                raise ValueError(f"非法输入维度：{rev_logit.shape}，预期二维或三维张量")

          
            labels = labels.view(-1)
            if labels.dim() > 1:
                labels = labels.squeeze()

            
            if rev_logit.size(0) != labels.size(0):
                
                if rev_logit.size(0) == 1 and labels.size(0) > 1:
                    rev_logit = rev_logit.expand(labels.size(0), -1)
                else:
                    raise AssertionError(
                        f"维度不匹配：输入样本数{rev_logit.shape[0]} ≠ 目标样本数{labels.shape[0]}\n"
                        f"模型输入形状: {common_emb.shape} → 输出形状: {rev_logit.shape}\n"
                        f"建议检查：1.模型最后一层的输出维度 2.标签是否包含序列维度"
                    )

            
            revloss = F.cross_entropy(
                rev_logit,
                labels,
                weight=labels_weight_balance
               
            )
            
            prob_all = F.log_softmax(logits_all, -1).view(-1, self.label_num)
            prob_kw = F.log_softmax(logits_kw, -1).view(-1, self.label_num)
            prob_detail = F.log_softmax(logits_detail, -1).view(-1, self.label_num)
            prob_joint = prob_kw.unsqueeze(-1).expand(-1, self.label_num, self.label_num) + \
                         prob_detail.unsqueeze(-2).expand(-1, self.label_num, self.label_num)
            prob_joint_list = []
            for idx in range(self.label_num):
                prob_dim = prob_joint[:, idx:, idx:].exp().sum((1, 2)) - prob_joint[:, idx + 1:, idx + 1:].exp().sum(
                    (1, 2))
                prob_joint_list.append(prob_dim.unsqueeze(-1))
            prob_kw_detail = (torch.cat(prob_joint_list, -1) + 1e-20).log()
            cls_loss = F.cross_entropy(logits_all.view(-1, self.label_num), labels.view(-1),
                                       weight=labels_weight_balance)
            kw_loss = F.cross_entropy(logits_kw.view(-1, self.label_num), labels.view(-1), weight=labels_weight_balance)
            kl_loss = 0.5 * (
                    F.kl_div(prob_kw_detail, prob_all, reduction='batchmean', log_target=True) +
                    F.kl_div(prob_all, prob_kw_detail, reduction='batchmean', log_target=True)
            )
            

            
            loss = cls_loss + revloss + kw_loss + kl_loss
            

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits_all
                
            )

        

        # encoding
        output_all = self.encoder(input_ids, attention_mask, token_type_ids, return_dict=True)
        

        # cls logits
        if "pooler_output" in output_all.keys():
            
            
            common_emb = dbma_attention(self.code(torch.zeros(1, dtype=torch.long).cuda()), sequence_out.detach(),
                                        sequence_out.detach(), attention_mask, dropout=None,
                                        block_size=64, num_global=4,
                                        expansion_ratio=0.5)
            
            sentence_proj = emb_proj(output_all.pooler_output, common_emb)
           
            sentence_embeddings=emb_proj(output_all.pooler_output,output_all.pooler_output-sentence_proj)
            logits_kw = self.classifier(self.dropout(sentence_embeddings))
            
            logits_all = self.classifier(self.dropout(output_all.pooler_output))
            
            logits_detail = self.classifier(self.dropout(sentence_proj))
            
        else:
            pooler_out_all = self.pooler(output_all.last_hidden_state[:, 0])
            
            sequence_out = output_all.last_hidden_state
            
            common_emb = dbma_attention(self.code(torch.zeros(1, dtype=torch.long).cuda()), sequence_out.detach(),
                                        sequence_out.detach(), attention_mask, dropout=None,
                                        block_size=64, num_global=4,
                                        expansion_ratio=0.5)
            sentence_proj = emb_proj(pooler_out_all , common_emb)
            sentence_embeddings = emb_proj(pooler_out_all, pooler_out_all - sentence_proj)
            logits_kw = self.classifier(self.dropout(sentence_embeddings))
            

            
            
            logits_all = self.classifier(self.dropout(pooler_out_all))
            
            logits_detail = self.classifier(self.dropout(sentence_proj))
            

        # joint probability distribution
        prob_all = F.log_softmax(logits_all, -1).view(-1, self.label_num)

        prob_kw = F.log_softmax(logits_kw, -1).view(-1, self.label_num)
        
        prob_detail = F.log_softmax(logits_detail, -1).view(-1, self.label_num)

        
        prob_joint = prob_kw.unsqueeze(-1).expand(-1, self.label_num, self.label_num) + \
           prob_detail.unsqueeze(-2).expand(-1, self.label_num, self.label_num)
        prob_joint_list = []
        for idx in range(self.label_num):
            prob_dim = prob_joint[:, idx:, idx:].exp().sum((1, 2)) - prob_joint[:, idx+1:, idx+1:].exp().sum((1, 2))
            prob_joint_list.append(prob_dim.unsqueeze(-1))
        prob_kw_detail = (torch.cat(prob_joint_list, -1)+1e-20).log()


        
        # 获取所有可能的标签
        all_labels = set(labels.cpu())
        #print(type(labels))
        labels_np=np.array(list(all_labels))
        final_labels=np.unique(labels_np)
        labels_weight_np=compute_class_weight(class_weight="balanced",classes=final_labels,y=labels_np)
        labels_weight_list=labels_weight_np.tolist()
        final_labels_list = final_labels.tolist()
        #label_fact=[0,1,2,3]
        #label_fact = [0, 1]
        label_fact = [0, 1, 2]
        labels_weight_modern=[1e-5 for element in label_fact if element not in final_labels_list]
        labels_weight_list+=labels_weight_modern
        labels_weight_list_np=np.array(labels_weight_list)


        device=torch.device("cuda"if torch.cuda.is_available()else "cpu")
        labels_weight_balance=torch.tensor(labels_weight_list_np).to(device).float()
        labels_weight_balance=labels_weight_balance.float()

        rev_logit=self.attention_classifier(common_emb)
        #
        if rev_logit.dim() == 3:
            # 处理三维输入 [batch, seq_len, label_num]
            if rev_logit.size(1) != 1:  # 常规序列标注
                rev_logit = rev_logit.view(-1, self.label_num)
            else:  # 处理单元素序列的特殊情况
                rev_logit = rev_logit.squeeze(1)  # 降维到 [batch, label_num]

        elif rev_logit.dim() == 2:
            # 处理二维输入 [batch, label_num]
            if rev_logit.size(0) != labels.size(0):
                rev_logit = rev_logit.repeat_interleave(labels.size(0) // rev_logit.size(0), dim=0)
        else:
            raise ValueError(f"非法输入维度：{rev_logit.shape}，预期二维或三维张量")

        # ===== 标签处理模块 =====
        labels = labels.view(-1)
        if labels.dim() > 1:
            labels = labels.squeeze()

        # ===== 动态维度对齐 =====
        if rev_logit.size(0) != labels.size(0):
            # 自动填充逻辑（处理batch_size=1的特殊情况）
            if rev_logit.size(0) == 1 and labels.size(0) > 1:
                rev_logit = rev_logit.expand(labels.size(0), -1)
            else:
                raise AssertionError(
                    f"维度不匹配：输入样本数{rev_logit.shape[0]} ≠ 目标样本数{labels.shape[0]}\n"
                    f"模型输入形状: {common_emb.shape} → 输出形状: {rev_logit.shape}\n"
                    f"建议检查：1.模型最后一层的输出维度 2.标签是否包含序列维度"
                )

        # ===== 交叉熵计算 =====
        revloss = F.cross_entropy(
            rev_logit,
            labels,
            weight=labels_weight_balance
           
        )
        
        kl_loss = 0.5 * (
                F.kl_div(prob_kw_detail, prob_all, reduction='batchmean', log_target=True) +
                F.kl_div(prob_all, prob_kw_detail, reduction='batchmean', log_target=True)
        )
        
        
        
        if self.loss_type == 1:

            loss = cls_loss + revloss+kw_loss+kl_loss
           
        
        if self.training:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits_all
                
            )
        
