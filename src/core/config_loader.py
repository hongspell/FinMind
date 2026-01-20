"""
Configuration Loader System
============================
Loads, validates, and manages all YAML configurations with hot-reload support.
Supports methodologies, agents, chains, prompts, and data sources.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import asyncio
from datetime import datetime
import logging
from functools import lru_cache
import re

logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """Types of configuration files"""
    METHODOLOGY = "methodologies"
    AGENT = "agents"
    CHAIN = "chains"
    PROMPT = "prompts"
    DATA_SOURCE = "data_sources"
    MODEL = "models"
    RISK = "risk"
    OUTPUT = "outputs"


@dataclass
class ConfigMetadata:
    """Metadata about a loaded configuration"""
    path: Path
    config_type: ConfigType
    name: str
    version: str
    last_loaded: datetime
    file_hash: str
    dependencies: List[str] = field(default_factory=list)
    

@dataclass
class ValidationError:
    """Represents a configuration validation error"""
    field: str
    message: str
    severity: str = "error"  # error, warning, info
    suggestion: Optional[str] = None


class ConfigValidator(ABC):
    """Base class for configuration validators"""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any], config_type: ConfigType) -> List[ValidationError]:
        """Validate a configuration and return list of errors"""
        pass


class SchemaValidator(ConfigValidator):
    """Validates configs against JSON schemas"""
    
    def __init__(self, schema_dir: Path):
        self.schema_dir = schema_dir
        self._schemas: Dict[ConfigType, Dict] = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load all JSON schemas"""
        schema_files = {
            ConfigType.METHODOLOGY: "methodology_schema.json",
            ConfigType.AGENT: "agent_schema.json",
            ConfigType.CHAIN: "chain_schema.json",
            ConfigType.PROMPT: "prompt_schema.json",
            ConfigType.DATA_SOURCE: "data_source_schema.json",
            ConfigType.MODEL: "model_schema.json",
        }
        
        for config_type, filename in schema_files.items():
            schema_path = self.schema_dir / filename
            if schema_path.exists():
                with open(schema_path) as f:
                    self._schemas[config_type] = json.load(f)
    
    def validate(self, config: Dict[str, Any], config_type: ConfigType) -> List[ValidationError]:
        """Validate config against schema"""
        errors = []
        
        schema = self._schemas.get(config_type)
        if not schema:
            # No schema defined, skip validation
            return errors
        
        # Basic required field validation
        required = schema.get("required", [])
        for field in required:
            if field not in config:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    suggestion=f"Add '{field}' to your configuration"
                ))
        
        # Type validation
        properties = schema.get("properties", {})
        for field, value in config.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if not self._check_type(value, expected_type):
                    errors.append(ValidationError(
                        field=field,
                        message=f"Field '{field}' has wrong type. Expected {expected_type}",
                        severity="error"
                    ))
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)


class SemanticValidator(ConfigValidator):
    """Validates semantic correctness of configurations"""
    
    def validate(self, config: Dict[str, Any], config_type: ConfigType) -> List[ValidationError]:
        """Validate semantic correctness"""
        errors = []
        
        if config_type == ConfigType.AGENT:
            errors.extend(self._validate_agent(config))
        elif config_type == ConfigType.CHAIN:
            errors.extend(self._validate_chain(config))
        elif config_type == ConfigType.METHODOLOGY:
            errors.extend(self._validate_methodology(config))
        
        return errors
    
    def _validate_agent(self, config: Dict) -> List[ValidationError]:
        """Validate agent configuration semantics"""
        errors = []
        
        # Check confidence factors sum to ~100%
        confidence_factors = config.get("confidence_factors", {})
        if confidence_factors:
            total_weight = sum(
                f.get("weight", 0) 
                for f in confidence_factors.values() 
                if isinstance(f, dict)
            )
            if abs(total_weight - 1.0) > 0.01:
                errors.append(ValidationError(
                    field="confidence_factors",
                    message=f"Confidence factor weights sum to {total_weight:.2f}, should be 1.0",
                    severity="warning",
                    suggestion="Adjust weights so they sum to 1.0"
                ))
        
        # Check guardrails are actionable
        guardrails = config.get("guardrails", [])
        vague_words = ["maybe", "possibly", "might", "could"]
        for i, guardrail in enumerate(guardrails):
            if isinstance(guardrail, str):
                if any(word in guardrail.lower() for word in vague_words):
                    errors.append(ValidationError(
                        field=f"guardrails[{i}]",
                        message="Guardrail contains vague language",
                        severity="warning",
                        suggestion="Use definitive language in guardrails"
                    ))
        
        return errors
    
    def _validate_chain(self, config: Dict) -> List[ValidationError]:
        """Validate chain configuration semantics"""
        errors = []
        
        stages = config.get("stages", [])
        stage_names = set()
        
        for i, stage in enumerate(stages):
            name = stage.get("name", f"stage_{i}")
            
            # Check for duplicate stage names
            if name in stage_names:
                errors.append(ValidationError(
                    field=f"stages[{i}].name",
                    message=f"Duplicate stage name: {name}",
                    severity="error"
                ))
            stage_names.add(name)
            
            # Check dependencies reference valid stages
            depends_on = stage.get("depends_on", [])
            for dep in depends_on:
                if dep not in stage_names:
                    errors.append(ValidationError(
                        field=f"stages[{i}].depends_on",
                        message=f"Stage '{name}' depends on unknown stage '{dep}'",
                        severity="error",
                        suggestion=f"Ensure '{dep}' is defined before '{name}'"
                    ))
            
            # Check for circular dependencies
            # (simplified - full check would need graph traversal)
            if name in depends_on:
                errors.append(ValidationError(
                    field=f"stages[{i}].depends_on",
                    message=f"Stage '{name}' has circular dependency on itself",
                    severity="error"
                ))
        
        # Check timeouts are reasonable
        timeout = config.get("timeout", {})
        total = timeout.get("total", 0)
        if total > 600:  # 10 minutes
            errors.append(ValidationError(
                field="timeout.total",
                message=f"Total timeout of {total}s is very long",
                severity="warning",
                suggestion="Consider breaking into smaller chains"
            ))
        
        return errors
    
    def _validate_methodology(self, config: Dict) -> List[ValidationError]:
        """Validate methodology configuration semantics"""
        errors = []
        
        # Check that calculation steps reference valid inputs
        steps = config.get("calculation_steps", [])
        available_vars = set(config.get("inputs", {}).keys())
        
        for i, step in enumerate(steps):
            step_name = step.get("name", f"step_{i}")
            inputs_used = step.get("inputs", [])
            
            for inp in inputs_used:
                if inp not in available_vars:
                    errors.append(ValidationError(
                        field=f"calculation_steps[{i}].inputs",
                        message=f"Step '{step_name}' uses undefined input '{inp}'",
                        severity="error"
                    ))
            
            # Add outputs to available vars for next steps
            outputs = step.get("outputs", [])
            available_vars.update(outputs)
        
        return errors


class ConfigLoader:
    """
    Main configuration loader with caching, validation, and hot-reload support.
    
    Usage:
        loader = ConfigLoader(config_dir="config/")
        
        # Load specific config
        agent_config = loader.load("valuation_agent", ConfigType.AGENT)
        
        # Load all configs of a type
        all_agents = loader.load_all(ConfigType.AGENT)
        
        # Enable hot reload
        loader.enable_hot_reload()
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path],
        schema_dir: Optional[Union[str, Path]] = None,
        enable_validation: bool = True
    ):
        self.config_dir = Path(config_dir)
        self.schema_dir = Path(schema_dir) if schema_dir else self.config_dir / "schemas"
        self.enable_validation = enable_validation
        
        # Cache for loaded configs
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, ConfigMetadata] = {}
        
        # Validators
        self._validators: List[ConfigValidator] = []
        if enable_validation:
            self._validators.append(SchemaValidator(self.schema_dir))
            self._validators.append(SemanticValidator())
        
        # Hot reload state
        self._hot_reload_enabled = False
        self._file_hashes: Dict[str, str] = {}
        self._reload_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Variable interpolation pattern
        self._var_pattern = re.compile(r'\$\{([^}]+)\}')
        
        logger.info(f"ConfigLoader initialized with config_dir: {self.config_dir}")
    
    def _get_config_path(self, name: str, config_type: ConfigType) -> Path:
        """Get the path for a configuration file"""
        type_dir = self.config_dir / config_type.value
        
        # Try different file extensions
        for ext in ['.yaml', '.yml', '.json']:
            path = type_dir / f"{name}{ext}"
            if path.exists():
                return path
        
        # Also check subdirectories
        for subdir in type_dir.iterdir():
            if subdir.is_dir():
                for ext in ['.yaml', '.yml', '.json']:
                    path = subdir / f"{name}{ext}"
                    if path.exists():
                        return path
        
        raise FileNotFoundError(
            f"Configuration '{name}' not found in {type_dir}"
        )
    
    def _compute_hash(self, path: Path) -> str:
        """Compute MD5 hash of a file"""
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _interpolate_variables(
        self, 
        config: Dict[str, Any], 
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpolate ${VAR} style variables in config values.
        Supports environment variables and custom variables.
        """
        if variables is None:
            variables = {}
        
        def interpolate_value(value: Any) -> Any:
            if isinstance(value, str):
                def replace_var(match):
                    var_name = match.group(1)
                    # Check custom variables first
                    if var_name in variables:
                        return str(variables[var_name])
                    # Then environment variables
                    env_val = os.environ.get(var_name)
                    if env_val is not None:
                        return env_val
                    # Return original if not found
                    return match.group(0)
                
                return self._var_pattern.sub(replace_var, value)
            elif isinstance(value, dict):
                return {k: interpolate_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [interpolate_value(item) for item in value]
            return value
        
        return interpolate_value(config)
    
    def _resolve_inheritance(self, config: Dict[str, Any], config_type: ConfigType) -> Dict[str, Any]:
        """
        Resolve config inheritance using 'extends' field.
        Child config overrides parent values.
        """
        extends = config.get("extends")
        if not extends:
            return config
        
        # Load parent config
        parent = self.load(extends, config_type)
        
        # Deep merge parent and child
        merged = self._deep_merge(parent, config)
        
        # Remove extends field from result
        merged.pop("extends", None)
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def load(
        self,
        name: str,
        config_type: ConfigType,
        variables: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Load a configuration by name and type.
        
        Args:
            name: Configuration name (without extension)
            config_type: Type of configuration
            variables: Custom variables for interpolation
            skip_cache: Force reload from disk
            
        Returns:
            Loaded and processed configuration dictionary
        """
        cache_key = f"{config_type.value}:{name}"
        
        # Check cache
        if not skip_cache and cache_key in self._cache:
            config = self._cache[cache_key]
            # Re-interpolate with new variables if provided
            if variables:
                return self._interpolate_variables(config, variables)
            return config
        
        # Find and load file
        path = self._get_config_path(name, config_type)
        file_hash = self._compute_hash(path)
        
        with open(path) as f:
            if path.suffix == '.json':
                raw_config = json.load(f)
            else:
                raw_config = yaml.safe_load(f)
        
        # Resolve inheritance
        config = self._resolve_inheritance(raw_config, config_type)
        
        # Interpolate variables
        config = self._interpolate_variables(config, variables)
        
        # Validate
        if self.enable_validation:
            errors = self.validate(config, config_type)
            error_count = sum(1 for e in errors if e.severity == "error")
            if error_count > 0:
                error_msgs = "\n".join(f"  - {e.field}: {e.message}" for e in errors if e.severity == "error")
                raise ValueError(f"Configuration validation failed for {name}:\n{error_msgs}")
            
            # Log warnings
            for error in errors:
                if error.severity == "warning":
                    logger.warning(f"Config warning in {name}: {error.field} - {error.message}")
        
        # Update cache and metadata
        self._cache[cache_key] = config
        self._metadata[cache_key] = ConfigMetadata(
            path=path,
            config_type=config_type,
            name=name,
            version=config.get("version", "1.0"),
            last_loaded=datetime.now(),
            file_hash=file_hash,
            dependencies=config.get("dependencies", [])
        )
        self._file_hashes[str(path)] = file_hash
        
        logger.debug(f"Loaded config: {cache_key} from {path}")
        
        return config
    
    def load_all(self, config_type: ConfigType) -> Dict[str, Dict[str, Any]]:
        """Load all configurations of a given type"""
        type_dir = self.config_dir / config_type.value
        configs = {}
        
        if not type_dir.exists():
            logger.warning(f"Config directory not found: {type_dir}")
            return configs
        
        # Find all config files
        for path in type_dir.rglob("*"):
            if path.suffix in ['.yaml', '.yml', '.json'] and path.is_file():
                name = path.stem
                try:
                    configs[name] = self.load(name, config_type)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
        
        return configs
    
    def validate(self, config: Dict[str, Any], config_type: ConfigType) -> List[ValidationError]:
        """Run all validators on a configuration"""
        all_errors = []
        for validator in self._validators:
            errors = validator.validate(config, config_type)
            all_errors.extend(errors)
        return all_errors
    
    def get_metadata(self, name: str, config_type: ConfigType) -> Optional[ConfigMetadata]:
        """Get metadata for a loaded configuration"""
        cache_key = f"{config_type.value}:{name}"
        return self._metadata.get(cache_key)
    
    def invalidate_cache(self, name: Optional[str] = None, config_type: Optional[ConfigType] = None):
        """Invalidate cache entries"""
        if name and config_type:
            cache_key = f"{config_type.value}:{name}"
            self._cache.pop(cache_key, None)
            self._metadata.pop(cache_key, None)
        elif config_type:
            prefix = f"{config_type.value}:"
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._metadata.pop(key, None)
        else:
            self._cache.clear()
            self._metadata.clear()
    
    def on_reload(self, callback: Callable[[str, Dict], None]):
        """Register a callback for config reloads"""
        self._reload_callbacks.append(callback)
    
    async def enable_hot_reload(self, check_interval: float = 5.0):
        """Enable hot reload with periodic file change detection"""
        self._hot_reload_enabled = True
        
        while self._hot_reload_enabled:
            await asyncio.sleep(check_interval)
            await self._check_for_changes()
    
    def disable_hot_reload(self):
        """Disable hot reload"""
        self._hot_reload_enabled = False
    
    async def _check_for_changes(self):
        """Check for file changes and reload if needed"""
        for cache_key, metadata in list(self._metadata.items()):
            try:
                current_hash = self._compute_hash(metadata.path)
                if current_hash != metadata.file_hash:
                    logger.info(f"Config changed: {metadata.name}, reloading...")
                    
                    # Reload
                    config = self.load(
                        metadata.name, 
                        metadata.config_type, 
                        skip_cache=True
                    )
                    
                    # Notify callbacks
                    for callback in self._reload_callbacks:
                        try:
                            callback(cache_key, config)
                        except Exception as e:
                            logger.error(f"Reload callback error: {e}")
                            
            except Exception as e:
                logger.error(f"Error checking config {cache_key}: {e}")


class MethodologyLoader:
    """
    Specialized loader for financial methodologies.
    Handles methodology-specific logic and formula parsing.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self._methodology_cache: Dict[str, 'LoadedMethodology'] = {}
    
    def load(self, name: str, category: str = "valuation") -> 'LoadedMethodology':
        """
        Load a methodology configuration.
        
        Args:
            name: Methodology name (e.g., 'dcf', 'comparable_companies')
            category: Methodology category (e.g., 'valuation', 'technical')
        """
        full_name = f"{category}/{name}" if category else name
        
        if full_name in self._methodology_cache:
            return self._methodology_cache[full_name]
        
        config = self.config_loader.load(full_name, ConfigType.METHODOLOGY)
        
        methodology = LoadedMethodology(
            name=name,
            category=category,
            config=config,
            inputs=config.get("inputs", {}),
            outputs=config.get("outputs", {}),
            calculation_steps=config.get("calculation_steps", []),
            sensitivity_variables=config.get("sensitivity_analysis", {}).get("variables", []),
            confidence_factors=config.get("confidence_factors", {}),
            warnings=config.get("warnings", [])
        )
        
        self._methodology_cache[full_name] = methodology
        return methodology


@dataclass
class LoadedMethodology:
    """Represents a loaded and parsed methodology"""
    name: str
    category: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    calculation_steps: List[Dict]
    sensitivity_variables: List[str]
    confidence_factors: Dict[str, Any]
    warnings: List[Dict]
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs"""
        return [
            name for name, spec in self.inputs.items()
            if spec.get("required", True)
        ]
    
    def get_optional_inputs(self) -> List[str]:
        """Get list of optional inputs with defaults"""
        return [
            name for name, spec in self.inputs.items()
            if not spec.get("required", True)
        ]
    
    def get_default_value(self, input_name: str) -> Any:
        """Get default value for an input"""
        spec = self.inputs.get(input_name, {})
        return spec.get("default")
    
    def validate_inputs(self, provided_inputs: Dict[str, Any]) -> List[str]:
        """Validate that all required inputs are provided"""
        missing = []
        for name in self.get_required_inputs():
            if name not in provided_inputs:
                missing.append(name)
        return missing


class PromptLoader:
    """
    Specialized loader for prompt templates.
    Supports Jinja2-style templating and prompt versioning.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self._prompt_cache: Dict[str, str] = {}
    
    def load(
        self, 
        name: str, 
        variables: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Load and render a prompt template.
        
        Args:
            name: Prompt name
            variables: Variables to interpolate
            version: Specific version to load (default: latest)
        """
        # Load prompt config
        config = self.config_loader.load(name, ConfigType.PROMPT)
        
        # Get template (support versioning)
        if version:
            template = config.get("versions", {}).get(version, {}).get("template")
            if not template:
                raise ValueError(f"Prompt version '{version}' not found")
        else:
            template = config.get("template", "")
        
        # Get system prompt if separate
        system_prompt = config.get("system_prompt", "")
        
        # Render template with variables
        if variables:
            template = self._render_template(template, variables)
            system_prompt = self._render_template(system_prompt, variables)
        
        # Return combined or just template
        if system_prompt:
            return {"system": system_prompt, "user": template}
        return template
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Simple template rendering with {{variable}} syntax"""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result
    
    def get_prompt_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata about a prompt"""
        config = self.config_loader.load(name, ConfigType.PROMPT)
        return {
            "name": name,
            "description": config.get("description", ""),
            "variables": config.get("variables", []),
            "versions": list(config.get("versions", {}).keys()),
            "default_model": config.get("default_model"),
            "temperature": config.get("temperature", 0.7)
        }


# Convenience function for creating a fully configured loader
def create_config_loader(
    config_dir: str = "config/",
    enable_validation: bool = True,
    enable_hot_reload: bool = False
) -> ConfigLoader:
    """
    Create a configured ConfigLoader instance.
    
    Args:
        config_dir: Path to configuration directory
        enable_validation: Enable config validation
        enable_hot_reload: Enable automatic config reloading
    """
    loader = ConfigLoader(
        config_dir=config_dir,
        enable_validation=enable_validation
    )
    
    if enable_hot_reload:
        # Start hot reload in background
        asyncio.create_task(loader.enable_hot_reload())
    
    return loader


# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    # Create test config
    test_config = {
        "name": "test_agent",
        "version": "1.0",
        "persona": {
            "role": "Test Analyst",
            "expertise": ["testing"]
        },
        "confidence_factors": {
            "data_quality": {"weight": 0.5},
            "completeness": {"weight": 0.5}
        },
        "guardrails": [
            "Always verify data",
            "Never make absolute claims"
        ]
    }
    
    # Test with temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        agents_dir = config_dir / "agents"
        agents_dir.mkdir()
        
        # Write test config
        with open(agents_dir / "test_agent.yaml", 'w') as f:
            yaml.dump(test_config, f)
        
        # Test loader
        loader = ConfigLoader(config_dir)
        loaded = loader.load("test_agent", ConfigType.AGENT)
        
        print("Loaded config:")
        print(json.dumps(loaded, indent=2))
        
        print("\nValidation passed!")
