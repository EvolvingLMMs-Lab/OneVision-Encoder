"""Tests for transformers 5.0+ compatibility with auto_map."""

import pytest
import tempfile
import shutil
import os
import json

from onevision_encoder import OneVisionEncoderConfig


class TestAutoMapConfiguration:
    """Tests for auto_map configuration required by transformers 5.0+."""

    def test_config_has_auto_map(self):
        """Test that config has auto_map for transformers 5.0+ compatibility."""
        config = OneVisionEncoderConfig()
        
        # Check auto_map exists
        assert hasattr(config, 'auto_map'), "Config should have auto_map attribute"
        assert isinstance(config.auto_map, dict), "auto_map should be a dictionary"
        
        # Check auto_map contains required keys
        assert 'AutoModel' in config.auto_map, "auto_map should contain AutoModel"
        assert 'AutoConfig' in config.auto_map, "auto_map should contain AutoConfig"
        
        # Check auto_map values
        assert config.auto_map['AutoModel'] == 'modeling_onevision_encoder.OneVisionEncoderModel'
        assert config.auto_map['AutoConfig'] == 'configuration_onevision_encoder.OneVisionEncoderConfig'

    def test_auto_map_preserved_with_custom_params(self):
        """Test that auto_map is present even with custom parameters."""
        config = OneVisionEncoderConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
        )
        
        assert hasattr(config, 'auto_map'), "Should have auto_map even with custom params"
        assert 'AutoModel' in config.auto_map
        assert 'AutoConfig' in config.auto_map

    def test_config_save_includes_auto_map(self):
        """Test that saved config includes auto_map."""
        config = OneVisionEncoderConfig(hidden_size=512)
        
        # Save to temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            config.save_pretrained(temp_dir)
            
            # Check saved config contains auto_map
            config_path = os.path.join(temp_dir, 'config.json')
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            
            assert 'auto_map' in saved_config, "Saved config should contain auto_map"
            assert 'AutoModel' in saved_config['auto_map']
            assert 'AutoConfig' in saved_config['auto_map']
            assert saved_config['hidden_size'] == 512, "Custom parameter should be saved"
        finally:
            # Clean up
            shutil.rmtree(temp_dir)

    def test_config_load_preserves_auto_map(self):
        """Test that loaded config preserves auto_map."""
        from transformers import AutoConfig
        
        # Create and save config
        config = OneVisionEncoderConfig(hidden_size=512)
        temp_dir = tempfile.mkdtemp()
        
        try:
            config.save_pretrained(temp_dir)
            
            # Copy Python files to temp directory for trust_remote_code
            import onevision_encoder
            src_dir = os.path.dirname(onevision_encoder.__file__)
            shutil.copy(os.path.join(src_dir, 'configuration_onevision_encoder.py'), temp_dir)
            shutil.copy(os.path.join(src_dir, 'modeling_onevision_encoder.py'), temp_dir)
            
            # Load config
            loaded_config = AutoConfig.from_pretrained(temp_dir, trust_remote_code=True)
            
            assert hasattr(loaded_config, 'auto_map'), "Loaded config should have auto_map"
            assert loaded_config.hidden_size == 512, "Custom parameter should be loaded"
            assert 'AutoModel' in loaded_config.auto_map
            assert 'AutoConfig' in loaded_config.auto_map
        finally:
            # Clean up
            shutil.rmtree(temp_dir)

    def test_auto_map_not_overridden_by_kwargs(self):
        """Test that auto_map from kwargs is respected."""
        custom_auto_map = {
            'AutoModel': 'custom_module.CustomModel',
            'AutoConfig': 'custom_module.CustomConfig',
        }
        
        config = OneVisionEncoderConfig(
            hidden_size=768,
            auto_map=custom_auto_map
        )
        
        # When auto_map is provided in kwargs, it should NOT be overridden
        # Our implementation checks if 'auto_map' not in kwargs before setting it
        assert hasattr(config, 'auto_map')
        # The config should use the provided auto_map
        assert config.auto_map == custom_auto_map


class TestTransformersVersionCompatibility:
    """Tests to verify compatibility across transformers versions."""

    def test_transformers_version_available(self):
        """Log current transformers version for debugging."""
        import transformers
        version = transformers.__version__
        
        print(f"\nTesting with transformers version: {version}")
        
        # Verify we can import required classes
        from transformers import AutoConfig, AutoModel
        from onevision_encoder import OneVisionEncoderConfig, OneVisionEncoderModel
        
        # Test passes if imports work
        assert True

    def test_config_works_with_current_transformers(self):
        """Test that config works with current transformers version."""
        config = OneVisionEncoderConfig()
        
        # Should be able to create config without errors
        assert config is not None
        assert hasattr(config, 'model_type')
        assert config.model_type == 'onevision_encoder'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
