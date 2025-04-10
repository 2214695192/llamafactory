# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import TYPE_CHECKING, Dict, Set, Optional, List, Literal, Tuple
import math

import torch
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from .model_utils.visual import get_forbidden_modules, patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _get_layer_type(name: str) -> Literal["attention", "mlp", "other"]:
    """Determine layer type based on common naming conventions."""
    # Attention-related modules (add more specific names if needed)
    if any(key in name.lower() for key in ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "output.dense"]):
        return "attention"
    # MLP/FeedForward-related modules (add more specific names if needed)
    elif any(key in name.lower() for key in ["mlp", "ffn", "feed_forward", "fc1", "fc2", "w1", "w2", "gate_proj", "up_proj", "down_proj", "intermediate.dense", "output.dense"]):
         # Note: "output.dense" could be ambiguous, prioritize Attention if pattern matches above
         # If name didn't match attention patterns, assume it's MLP if these keywords are present
         if "attention" not in name.lower() and "attn" not in name.lower():
              return "mlp"
    return "other"


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = True,
    cast_trainable_params_to_fp32: bool = False,
    output_dir: Optional[str] = None
) -> "PeftModel":
    """
    Setup LoRA tuning for the model.
    """
    if is_trainable:
        logger.info_rank0("Fine-tuning method: {}".format("Knowledge-Aware " + ("DoRA" if finetuning_args.use_dora else "LoRA")))

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        for adapter in adapter_to_merge:
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume lora training
            if model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, is_trainable=is_trainable)
            else:
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:
        target_modules = []
        rank_pattern_dict = {}

        # Handle target modules first
        if isinstance(finetuning_args.lora_target, str):
            if finetuning_args.lora_target == "all":
                target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
            else:
                target_modules = [module.strip() for module in finetuning_args.lora_target.split(",")]
        elif isinstance(finetuning_args.lora_target, (list, set)):
            if "all" in finetuning_args.lora_target:
                target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
            else:
                target_modules = list(finetuning_args.lora_target)
        else:
            target_modules = []

        if not target_modules:
            logger.warning("No target modules specified. Using all linear layers.")
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)

        # Calculate dynamic ranks only for target modules
        if finetuning_args.use_dynamic_rank:
            num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "num_layers", None)
            if not num_layers:
                raise ValueError("Cannot determine number of layers for dynamic rank allocation.")

            logger.info_rank0(
                f"Calculating dynamic ranks with pattern='{finetuning_args.rank_pattern}', "
                f"base_rank={finetuning_args.lora_rank}, min={finetuning_args.min_rank}, max={finetuning_args.max_rank}, "
                f"middle_factor={finetuning_args.middle_layer_factor}, attn_boost={finetuning_args.attention_boost_factor}, "
                f"smooth_factor={finetuning_args.rank_smooth_factor}, base_factor={finetuning_args.rank_base_factor}"
            )

            module_names_with_rank = set()
            for name, module in model.named_modules():
                module_class_name = module.__class__.__name__
                is_target_type = "Linear" in module_class_name or "LoraLayer" in module_class_name
                is_in_target_list = any(target == name or (target != "all" and target in name) for target in target_modules)

                if is_target_type and is_in_target_list:
                    layer_idx = _get_layer_idx(name)
                    layer_type = _get_layer_type(name)

                    if layer_idx is not None and layer_type != "other":
                        knowledge_score = _get_layer_knowledge_scores(
                            layer_idx=layer_idx,
                            total_layers=num_layers,
                            layer_type=layer_type,
                            layer_name=name,
                            pattern=finetuning_args.rank_pattern,
                            middle_factor=finetuning_args.middle_layer_factor,
                            attention_boost_factor=finetuning_args.attention_boost_factor
                        )
                        rank = _calculate_dynamic_rank(
                            base_rank=finetuning_args.lora_rank,
                            knowledge_score=knowledge_score,
                            min_rank=finetuning_args.min_rank,
                            max_rank=finetuning_args.max_rank,
                            smooth_factor=finetuning_args.rank_smooth_factor,
                            base_factor=finetuning_args.rank_base_factor
                        )
                        rank_pattern_dict[name] = rank
                        module_names_with_rank.add(name)
                    elif layer_idx is not None:
                        pass

            if not rank_pattern_dict:
                logger.warning("No valid layers (with identifiable type and index) found for dynamic rank allocation. Using default rank.")

        # Handle modules to save
        if isinstance(finetuning_args.lora_modules_to_save, str):
            modules_to_save = finetuning_args.lora_modules_to_save.split(",") if finetuning_args.lora_modules_to_save else None
        else:
            modules_to_save = finetuning_args.lora_modules_to_save

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetuning_args.lora_rank,
            lora_alpha=finetuning_args.lora_alpha,
            lora_dropout=finetuning_args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            rank_pattern=rank_pattern_dict if finetuning_args.use_dynamic_rank else {},
            use_rslora=finetuning_args.use_rslora,
            use_dora=finetuning_args.use_dora
        )

        if model_args.use_unsloth:
            model = get_unsloth_peft_model(config, model, lora_config, model_args)
        else:
            model = get_peft_model(model, lora_config)

        if modules_to_save is not None:
            logger.info_rank0("Modules to save: {}".format(",".join(modules_to_save)))

        if finetuning_args.use_dynamic_rank:
            if rank_pattern_dict:
                if len(rank_pattern_dict) > 10:
                    rank_summary = {k: v for i, (k, v) in enumerate(rank_pattern_dict.items()) if i < 5}
                    rank_summary["..."] = f"({len(rank_pattern_dict)} layers assigned)"
                    logger.info_rank0(f"Applied dynamic rank (sample): {rank_summary}")
                else:
                    logger.info_rank0(f"Applied dynamic rank: {rank_pattern_dict}")
            else:
                logger.warning("Dynamic rank enabled but no ranks were assigned.")

    return model


def _get_layer_idx(name: str) -> Optional[int]:
    """Extract layer index from parameter name."""
    match = re.search(r'\.(\d+)\.', name)
    if match:
        return int(match.group(1))
    match_block = re.search(r'block_(\d+)', name)
    if match_block:
        return int(match_block.group(1))
    return None


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantized models can only be used for the LoRA tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 and not fsdp (zero3 or fsdp already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and (is_deepspeed_zero3_enabled() or is_fsdp_enabled()):
        logger.info_rank0("ZeRO3 / FSDP detected, remaining trainable params in float32.")
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")

    return model


def get_trainer(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"],
    callbacks: Optional[List["TrainerCallback"]] = None,
) -> "Trainer":
    # ... existing code ...
    
    if finetuning_args.finetuning_type == "lora" and finetuning_args.use_dynamic_rank:
        trainer = KnowledgeAwareTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    
    return trainer


def _get_layer_knowledge_scores(
    layer_idx: int,
    total_layers: int,
    layer_type: Literal["attention", "mlp"],
    layer_name: str,
    pattern: str = "gaussian",
    middle_factor: float = 1.0,
    attention_boost_factor: float = 1.5
) -> float:
    """
    Calculate knowledge importance scores based on layer position and type.

    Args:
        layer_idx: Index of the current layer (0-based).
        total_layers: Total number of layers in the model.
        layer_type: Type of the layer ('attention' or 'mlp').
        layer_name: Full name of the layer.
        pattern: Base pattern for positional scoring ("gaussian", "linear", "constant", "late_bias", "early_bias").
        middle_factor: Base factor for middle layers in gaussian/linear patterns.
        attention_boost_factor: Multiplier for attention layers in the middle block.

    Returns:
        float: Knowledge importance score for the layer.
    """
    if total_layers <= 1: return 1.0
    normalized_pos = layer_idx / (total_layers - 1) # [0, 1]

    # 1. 计算基础的位置得分 (base_score)
    base_score = 1.0 # Default for constant or unknown pattern
    mean = 0.5
    std_dev = 0.15
    boost_width = 0.25 # 定义中间区域 +/- mean (0.25 * 2 = 50% of layers)

    if pattern == "gaussian":
        base_score = math.exp(-((normalized_pos - mean) ** 2) / (2 * std_dev ** 2))
        boost_factor = max(1.0, middle_factor)
        if abs(normalized_pos - mean) <= boost_width:
             cos_scaling = (math.cos(math.pi * (normalized_pos - mean) / boost_width) + 1) / 2
             base_score *= 1 + (boost_factor - 1) * cos_scaling
    elif pattern == "linear":
        base_score = 1.0 - 2 * abs(normalized_pos - mean) # V-shape
        boost_factor = max(1.0, middle_factor)
        if abs(normalized_pos - mean) <= boost_width:
             cos_scaling = (math.cos(math.pi * (normalized_pos - mean) / boost_width) + 1) / 2
             base_score *= 1 + (boost_factor - 1) * cos_scaling
    elif pattern == "late_bias":
        base_score = 0.1 + 0.9 * normalized_pos
    elif pattern == "early_bias":
        base_score = 0.1 + 0.9 * (1.0 - normalized_pos)
    elif pattern == "constant":
        base_score = 1.0
    else:
        logger.warning(f"Unknown rank pattern: {pattern}. Using constant base score.")
        base_score = 1.0

    base_score = max(0.0, base_score) # Ensure non-negative base score

    # 2. 根据层类型和位置应用提升因子
    final_score = base_score
    # 定义中层区域 (例如，占总层数的 50%，从 25% 到 75%)
    middle_start_ratio = 0.25
    middle_end_ratio = 0.75
    is_middle_layer = (middle_start_ratio <= normalized_pos <= middle_end_ratio)

    if layer_type == "attention" and is_middle_layer:
        final_score *= max(1.0, attention_boost_factor) # 应用提升因子，确保不小于1

    return max(0.0, final_score) # 返回最终得分，确保非负


def _calculate_dynamic_rank(
    base_rank: int,
    knowledge_score: float,
    min_rank: int = 4,
    max_rank: int = 32,
    smooth_factor: float = 0.8,
    base_factor: float = 0.2
) -> int:
    """
    Calculate dynamic rank based on knowledge importance score.
    
    Args:
        base_rank: Base rank value
        knowledge_score: Knowledge importance score
        min_rank: Minimum rank value
        max_rank: Maximum rank value
        smooth_factor: Weight for the knowledge score component
        base_factor: Weight for the base component (ensures minimum)
        
    Returns:
        int: Calculated rank for the layer
    """
    if not (0 <= smooth_factor <= 1 and 0 <= base_factor <= 1 and smooth_factor + base_factor > 0):
        logger.warning(f"Invalid smoothing factors (smooth={smooth_factor}, base={base_factor}). Using default 0.8/0.2.")
        smooth_factor = 0.8
        base_factor = 0.2

    # Cap score to prevent extreme ranks, e.g., if score * boost > 1 significantly
    capped_score = max(0.0, min(knowledge_score, 2.5)) # 设定一个上限，例如 2.5

    effective_score = base_factor + smooth_factor * capped_score

    rank = int(base_rank * effective_score)

    final_rank = max(min_rank, min(rank, max_rank))

    return final_rank
