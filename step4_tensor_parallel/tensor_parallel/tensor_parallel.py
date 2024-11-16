"""
Inspired by Fair Scale/Megatron's Tensor Parallelism implementation
Ref: https://github.com/facebookresearch/fairscale/tree/main/fairscale
"""
import torch
import math
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Callable, Optional
import process_group_manager as pgm
from functools import partial
import torch.nn.init as init

from .tp_utils import VocabUtility
from .tp_communications import copy_to_model_parallel_region, gather_from_model_parallel_region, reduce_from_model_parallel_region

def apply_tensor_parallel(model, init_method):

    def _replace_module(_module, _linear_proj_name, _style, _init_method, args={}):
        assert _style in ["column", "row", 'vocab']
        linear_layer = getattr(_module, _linear_proj_name)
        
        if _style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                init_method=_init_method,
                gather_output=args.get("gather_output", False)
            )
        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                init_method=_init_method
            )
        else:
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
                init_method=partial(_init_method, vocab_embedding=True)
            )
        setattr(_module, _linear_proj_name, new_linear_layer)

    module_linear_name_stype_mapping_list = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]

    for layer in model.decoder_layers:
        for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
            _replace_module(getattr(layer, module_name), linear_proj_name, style, init_method)
            
    _replace_module(model, "embedding", "vocab", init_method)
    _replace_module(model, "final_proj", "column", init_method, args={"gather_output": True})
    
    return model

def initialize_weight_tensor(weight, vocab_embedding=False):
    """
    Initialize the weight tensor with the default initialization method in PyTorch
    If not a vocab embedding, it uses U(-sqrt(k), sqrt(k)) with k = 1/in_features.
    If it's a vocab embedding, it uses a normal distribution N(0, 1).
    """
    if not vocab_embedding:
        # Get the in_features from the shape of the weight tensor
        _, in_features = weight.shape
        
        # Calculate k and the uniform bounds
        k = 1 / in_features
        bound = math.sqrt(k)
        
        # Initialize weights with U(-sqrt(k), sqrt(k))
        torch.nn.init.uniform_(weight, -bound, bound)
    else:
        # Initialize Vocab embedding with N(0, 1)
        torch.nn.init.normal_(weight, mean=0.0, std=1.0)

def _initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[torch.Tensor], torch.Tensor]
) -> Optional[torch.Tensor]:
    """
    Initialize the master weights for the entire linear layer. Each process will take a partition of the master weight
    Args:
        weight: The weight tensor that will be initialized for the current partition.
        out_features: second dimension of weight matrix W.
        in_features: first dimension of weight matrix W.
        per_partition_size: The size of the weight partition assigned to each process.
        partition_dim: The dimension along which the weight matrix is split for parallelism.
        init_method: The method used to initialize the weight values.
    """

    # If we only use 1 process for model parallelism, we can simply initialize the weight
    if pgm.process_group_manager.tp_world_size == 1:
        init_method(weight)
        return None

    # Initialize master weight
    master_weight = torch.empty(out_features, in_features, dtype=weight.dtype, requires_grad=False)
    init_method(master_weight)
    
    # Split the model into size of per_partition_size and take the corresponding partition
    weight_list = torch.split(master_weight, per_partition_size, dim=partition_dim)
    weight.data = weight_list[pgm.process_group_manager.tp_rank].contiguous()

    return None

class ColumnParallelLinear(torch.nn.Module):
    """Column Parallel Linear layer
    Y = XW + b, where weight matrix W is parallelized along its second dimension. W = [W_1, ..., W_p]
    This module returns the results of Y_i = XW_i + b_i in the forward method, Y_i is parallelized in the second dimension.
    Arguments:
        in_features: first dimension of weight matrix W.
        out_features: second dimension of weight matrix W.
        bias: If true, add bias
        init_method: method to initialize weights
        gather_output: If true, gather the output from all the partitions. This is used for the last linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        gather_output: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % pgm.process_group_manager.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = out_features // pgm.process_group_manager.tp_world_size
        self.gather_output = gather_output

        # Allocate space for the weight and bias
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features)) # W_i
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            partition_dim = 0,
            init_method = init_method,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  
        input_parallel = copy_to_model_parallel_region(input_)
        output = F.linear(input_parallel, self.weight, self.bias) # XW_i^T + b, output is Y_i
        if self.gather_output:
            output = gather_from_model_parallel_region(output)
        return output
    
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.
    Y = XW + b. W is parallelized along its first dimension and X along its second dimension as:
               -   -
              | W_1 |
              | .   |
          W = | .   |        X = [X_1, ..., X_p]
              | .   |
              | W_p |
               -   -
    We assume that X is already parallelized. This is the case after ColumnParallelLinear.
    This module returns the results of Y = sum(X_i * W_i + b_i) in the forward method.
    Arguments:
        in_features: first dimension of matrix W.
        out_features: second dimension of matrix W.
        bias: If true, add bias
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_size_per_partition = in_features // pgm.process_group_manager.tp_world_size

        self.weight = Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            partition_dim = 1,
            init_method = init_method,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor: 
        output_parallel = F.linear(input_, self.weight)  # X_i * W_i^T + b
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output   
    
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    This is mainly adapted from torch.nn.Embedding and all the default values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
    ) -> None:
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim, self.num_embeddings_per_partition, 0, init_method
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Performs an embedding lookup for input tokens in the parallelized embedding layer
        1. Masks tokens that fall outside the specified vocabulary range and adjusts the input
        2. Performs embedding lookups for valid tokens, setting embeddings of out-of-vocabulary tokens to zero
        3. Reduces the embeddings across model parallel GPUs using all-reduce for synchronization
        """
        # Build the mask for out-of-vocabulary tokens.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        # Get the embeddings for the valid tokens.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Embedding of out-of-vocabulary tokens is set to 0.
        output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs to get the final output.
        output = reduce_from_model_parallel_region(output_parallel)
        return output