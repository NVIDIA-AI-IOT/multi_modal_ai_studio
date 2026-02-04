#!/usr/bin/env python3
"""
Verify Multi-modal AI Studio setup is complete.

Run this script to check that all components are in place and working.
"""

import sys
from pathlib import Path

# Add src to path for testing without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_structure():
    """Check directory structure is complete."""
    print("🔍 Checking directory structure...")
    
    required_dirs = [
        ".cursor/rules",
        "docs/cursor",
        "presets",
        "src/multi_modal_ai_studio/config",
        "src/multi_modal_ai_studio/backends/asr",
        "src/multi_modal_ai_studio/backends/llm",
        "src/multi_modal_ai_studio/backends/tts",
        "src/multi_modal_ai_studio/cli",
        "src/multi_modal_ai_studio/core",
        "src/multi_modal_ai_studio/devices",
        "src/multi_modal_ai_studio/server",
        "src/multi_modal_ai_studio/utils",
        "tests",
    ]
    
    root = Path(__file__).parent.parent
    missing = []
    
    for dir_path in required_dirs:
        if not (root / dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        print(f"  ❌ Missing directories: {', '.join(missing)}")
        return False
    
    print("  ✅ All required directories present")
    return True


def check_files():
    """Check key files exist."""
    print("\n🔍 Checking key files...")
    
    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "LICENSE",
        ".gitignore",
        ".cursor/index.mdc",
        "presets/default.yaml",
        "src/multi_modal_ai_studio/__init__.py",
        "src/multi_modal_ai_studio/config/schema.py",
        "src/multi_modal_ai_studio/cli/main.py",
        "docs/cursor/PLAN_MULTI_MODAL_AI_STUDIO.md",
    ]
    
    root = Path(__file__).parent.parent
    missing = []
    
    for file_path in required_files:
        if not (root / file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"  ❌ Missing files: {', '.join(missing)}")
        return False
    
    print("  ✅ All required files present")
    return True


def check_presets():
    """Check all presets are valid."""
    print("\n🔍 Checking presets...")
    
    try:
        from multi_modal_ai_studio.config import SessionConfig
        
        root = Path(__file__).parent.parent
        presets_dir = root / "presets"
        
        presets = list(presets_dir.glob("*.yaml"))
        if not presets:
            print("  ❌ No presets found")
            return False
        
        for preset_file in presets:
            try:
                cfg = SessionConfig.from_yaml(preset_file)
                warnings = cfg.validate()
                
                # Check for critical errors (not just warnings)
                has_errors = False
                for component, warns in warnings.items():
                    for warn in warns:
                        if "required" in warn.lower():
                            print(f"  ⚠️  {preset_file.name}: {component}: {warn}")
                            has_errors = True
                
                if not has_errors:
                    print(f"  ✅ {preset_file.name}: Valid")
            except Exception as e:
                print(f"  ❌ {preset_file.name}: Failed to load ({e})")
                return False
        
        return True
    
    except ImportError as e:
        print(f"  ❌ Cannot import config module: {e}")
        return False


def check_config_module():
    """Check config module is functional."""
    print("\n🔍 Checking config module...")
    
    try:
        from multi_modal_ai_studio.config import (
            SessionConfig, ASRConfig, LLMConfig, TTSConfig, DeviceConfig, AppConfig
        )
        
        print("  ✅ All config classes importable")
        
        # Test creating a config
        cfg = SessionConfig(
            name="Test",
            asr=ASRConfig(scheme="riva"),
            llm=LLMConfig(scheme="openai"),
            tts=TTSConfig(scheme="riva")
        )
        
        print(f"  ✅ Config creation works")
        print(f"     Name: {cfg.name}")
        print(f"     Services: {cfg.get_required_services()}")
        print(f"     Mode: {cfg.devices.get_mode_description()}")
        
        # Test serialization
        data = cfg.to_dict()
        cfg2 = SessionConfig.from_dict(data)
        print("  ✅ Serialization works")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Config module error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cli():
    """Check CLI module exists and is importable."""
    print("\n🔍 Checking CLI module...")
    
    try:
        from multi_modal_ai_studio.cli import main
        print("  ✅ CLI module importable")
        
        # Check if main function exists
        if hasattr(main, 'main'):
            print("  ✅ main() function present")
        else:
            print("  ⚠️  main() function not found (expected)")
        
        return True
    
    except ImportError as e:
        print(f"  ❌ Cannot import CLI: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Multi-modal AI Studio - Setup Verification")
    print("=" * 60)
    
    checks = [
        check_structure,
        check_files,
        check_config_module,
        check_presets,
        check_cli,
    ]
    
    results = [check() for check in checks]
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("  1. Install package: pip install -e .")
        print("  2. Start Phase 1: Port backends from live-riva-webui")
        print("  3. See docs/cursor/PLAN_MULTI_MODAL_AI_STUDIO.md")
        return 0
    else:
        print("❌ Some checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
