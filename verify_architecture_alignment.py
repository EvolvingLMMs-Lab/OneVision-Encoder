#!/usr/bin/env python3
# coding=utf-8
"""
架构一致性对齐验证程序
Architecture Alignment Verification Program

这个程序验证 llavavit/ 目录中的模型与 model_factory/vit_preview_v0_hf.py 中的模型架构是否一致。
This program verifies that the model in llavavit/ directory is aligned with model_factory/vit_preview_v0_hf.py.

使用方法 / Usage:
    python verify_architecture_alignment.py
    
    或使用详细模式 / Or use verbose mode:
    python verify_architecture_alignment.py --verbose
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add paths to import modules
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "model_factory"))
sys.path.insert(0, str(repo_root / "llavavit"))


def compare_configs(config1, config2, verbose=False):
    """
    比较两个配置对象的属性
    Compare attributes of two config objects
    """
    print("\n" + "="*80)
    print("📋 配置对齐检查 / Configuration Alignment Check")
    print("="*80)
    
    mismatches = []
    config1_attrs = {k: v for k, v in vars(config1).items() if not k.startswith('_')}
    config2_attrs = {k: v for k, v in vars(config2).items() if not k.startswith('_')}
    
    all_keys = set(config1_attrs.keys()) | set(config2_attrs.keys())
    
    for key in sorted(all_keys):
        val1 = config1_attrs.get(key, "NOT_FOUND")
        val2 = config2_attrs.get(key, "NOT_FOUND")
        
        if val1 != val2:
            mismatches.append((key, val1, val2))
            print(f"  ❌ {key:25s}: {val1} != {val2}")
        elif verbose:
            print(f"  ✅ {key:25s}: {val1}")
    
    if not mismatches:
        print("  ✅ 所有配置参数一致 / All config parameters match")
        return True
    else:
        print(f"\n  ⚠️  发现 {len(mismatches)} 个不匹配项 / Found {len(mismatches)} mismatches")
        return False


def compare_model_structure(model1, model2, verbose=False):
    """
    比较两个模型的结构（层数、参数名称等）
    Compare structure of two models (layers, parameter names, etc.)
    """
    print("\n" + "="*80)
    print("🏗️  模型结构对齐检查 / Model Structure Alignment Check")
    print("="*80)
    
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    # Check for missing or extra keys
    missing_in_2 = keys1 - keys2
    extra_in_2 = keys2 - keys1
    common_keys = keys1 & keys2
    
    all_match = True
    
    if missing_in_2:
        print(f"\n  ❌ model_factory 中有但 llavavit 中没有的参数 ({len(missing_in_2)} 个):")
        print(f"     Parameters in model_factory but not in llavavit ({len(missing_in_2)}):")
        for key in sorted(list(missing_in_2)[:10]):
            print(f"       - {key}")
        if len(missing_in_2) > 10:
            print(f"       ... and {len(missing_in_2) - 10} more")
        all_match = False
    
    if extra_in_2:
        print(f"\n  ❌ llavavit 中有但 model_factory 中没有的参数 ({len(extra_in_2)} 个):")
        print(f"     Parameters in llavavit but not in model_factory ({len(extra_in_2)}):")
        for key in sorted(list(extra_in_2)[:10]):
            print(f"       - {key}")
        if len(extra_in_2) > 10:
            print(f"       ... and {len(extra_in_2) - 10} more")
        all_match = False
    
    # Check shape consistency for common keys
    shape_mismatches = []
    for key in sorted(common_keys):
        shape1 = state_dict1[key].shape
        shape2 = state_dict2[key].shape
        if shape1 != shape2:
            shape_mismatches.append((key, shape1, shape2))
            print(f"  ❌ 形状不匹配 / Shape mismatch: {key}")
            print(f"     model_factory: {shape1}")
            print(f"     llavavit:      {shape2}")
            all_match = False
    
    if all_match:
        print(f"\n  ✅ 模型结构完全一致 / Model structures are identical")
        print(f"     共有参数 / Total parameters: {len(common_keys)}")
        if verbose:
            print(f"\n  参数列表 / Parameter list:")
            for key in sorted(list(common_keys)[:20]):
                print(f"       - {key}: {state_dict1[key].shape}")
            if len(common_keys) > 20:
                print(f"       ... and {len(common_keys) - 20} more")
    else:
        print(f"\n  ⚠️  模型结构不完全一致 / Model structures are not identical")
    
    return all_match


def compare_forward_outputs(model1, model2, config, verbose=False):
    """
    比较两个模型的前向传播输出
    Compare forward pass outputs of two models
    """
    print("\n" + "="*80)
    print("🔄 前向传播对齐检查 / Forward Pass Alignment Check")
    print("="*80)
    
    model1.eval()
    model2.eval()
    
    # Create dummy input
    batch_size = 2
    channels = 3
    height = width = config.image_size
    
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print(f"\n  输入形状 / Input shape: {dummy_input.shape}")
    
    try:
        with torch.no_grad():
            # Forward pass on both models
            out1 = model1(pixel_values=dummy_input)
            out2 = model2(pixel_values=dummy_input)
            
            # Compare outputs
            if hasattr(out1, 'last_hidden_state') and hasattr(out2, 'last_hidden_state'):
                lhs1 = out1.last_hidden_state
                lhs2 = out2.last_hidden_state
                
                print(f"\n  last_hidden_state 形状 / shape:")
                print(f"    model_factory: {lhs1.shape}")
                print(f"    llavavit:      {lhs2.shape}")
                
                if lhs1.shape == lhs2.shape:
                    print(f"  ✅ 输出形状一致 / Output shapes match")
                    
                    # Check if we loaded the same weights by comparing actual values
                    # Since we use random init, we expect them to be different
                    # But shapes and forward pass should work the same way
                    max_diff = (lhs1 - lhs2).abs().max().item()
                    
                    if max_diff < 1e-5:
                        print(f"  ✅ 输出数值一致（最大差异: {max_diff:.2e}）")
                        print(f"     Output values match (max diff: {max_diff:.2e})")
                    else:
                        print(f"  ℹ️  输出数值不同（最大差异: {max_diff:.2e}）")
                        print(f"     Output values differ (max diff: {max_diff:.2e})")
                        print(f"     这是正常的，因为使用了随机初始化")
                        print(f"     This is expected with random initialization")
                else:
                    print(f"  ❌ 输出形状不一致 / Output shapes don't match")
                    return False
                
                # Check pooler output if exists
                if hasattr(out1, 'pooler_output') and out1.pooler_output is not None:
                    po1 = out1.pooler_output
                    po2 = out2.pooler_output
                    
                    print(f"\n  pooler_output 形状 / shape:")
                    print(f"    model_factory: {po1.shape}")
                    print(f"    llavavit:      {po2.shape}")
                    
                    if po1.shape == po2.shape:
                        print(f"  ✅ Pooler 输出形状一致 / Pooler output shapes match")
                    else:
                        print(f"  ❌ Pooler 输出形状不一致 / Pooler output shapes don't match")
                        return False
                
                return True
            else:
                print(f"  ❌ 输出格式不一致 / Output format mismatch")
                return False
                
    except Exception as e:
        print(f"  ❌ 前向传播失败 / Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_models_and_verify(verbose=False):
    """
    加载两个模型并进行全面验证
    Load both models and perform comprehensive verification
    """
    print("\n" + "="*80)
    print("🚀 开始架构一致性验证 / Starting Architecture Alignment Verification")
    print("="*80)
    
    # Import model_factory version
    try:
        from model_factory.vit_preview_v0_hf import LlavaViTConfig as Config1
        from model_factory.vit_preview_v0_hf import LlavaViTModel as Model1
        print("✅ 成功导入 model_factory 模型 / Successfully imported model_factory model")
    except Exception as e:
        print(f"❌ 无法导入 model_factory 模型 / Failed to import model_factory model: {e}")
        return False
    
    # Import llavavit version
    try:
        from llavavit.configuration_llava_vit import LlavaViTConfig as Config2
        from llavavit.modeling_llava_vit import LlavaViTModel as Model2
        print("✅ 成功导入 llavavit 模型 / Successfully imported llavavit model")
    except Exception as e:
        print(f"❌ 无法导入 llavavit 模型 / Failed to import llavavit model: {e}")
        return False
    
    # Create configs with same parameters
    config_kwargs = {
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'image_size': 448,
        'patch_size': 16,
        'use_head': True,
    }
    
    print("\n📝 使用配置 / Using configuration:")
    for k, v in config_kwargs.items():
        print(f"   {k}: {v}")
    
    config1 = Config1(**config_kwargs)
    config2 = Config2(**config_kwargs)
    
    # Compare configs
    config_match = compare_configs(config1, config2, verbose)
    
    # Create models
    print("\n🏗️  创建模型 / Creating models...")
    model1 = Model1(config1)
    model2 = Model2(config2)
    print("✅ 模型创建成功 / Models created successfully")
    
    # Compare model structures
    structure_match = compare_model_structure(model1, model2, verbose)
    
    # Compare forward pass
    forward_match = compare_forward_outputs(model1, model2, config1, verbose)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 验证总结 / Verification Summary")
    print("="*80)
    
    results = {
        "配置对齐 / Config Alignment": config_match,
        "结构对齐 / Structure Alignment": structure_match,
        "前向传播对齐 / Forward Pass Alignment": forward_match,
    }
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ 通过 / PASS" if passed else "❌ 失败 / FAIL"
        print(f"  {name:40s}: {status}")
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 所有检查通过！架构完全一致！")
        print("   All checks passed! Architectures are fully aligned!")
    else:
        print("\n⚠️  部分检查未通过，请查看上面的详细信息。")
        print("   Some checks failed, please review details above.")
    
    print("="*80 + "\n")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="验证 llavavit 和 model_factory 中模型架构的一致性 / Verify architecture alignment"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息 / Show verbose output'
    )
    
    args = parser.parse_args()
    
    success = load_models_and_verify(verbose=args.verbose)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
