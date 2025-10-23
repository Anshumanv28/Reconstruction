# Plugin System Documentation

## Overview
The backend now features a fully pluggable model system that allows you to easily add, remove, and manage different layout detection and OCR models without modifying the core application code.

## Architecture

### Core Components

1. **Base Model Classes** (`models/base_model.py`)
   - `BaseModel`: Abstract base class for all models
   - `BaseLayoutModel`: Abstract base class for layout detection models
   - `BaseOCRModel`: Abstract base class for OCR models

2. **Model Factory** (`models/model_factory.py`)
   - Creates model instances dynamically
   - Validates model types and configurations
   - Handles model class loading

3. **Plugin Registry** (`models/plugin_registry.py`)
   - Manages model configurations
   - Loads/saves configuration files
   - Handles model discovery

4. **Model Registry** (`models/registry_new.py`)
   - Global registry for model instances
   - Manages model lifecycle (load/unload)
   - Provides unified interface

## Adding New Models

### Step 1: Create Your Model Class

Create a new model class that inherits from the appropriate base class:

```python
# models/layout/my_layout_model.py
from typing import Dict, Any, Optional
from PIL import Image
from ..base_model import BaseLayoutModel

class MyLayoutModel(BaseLayoutModel):
    def __init__(self, model_name: str = "my_layout_model", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        # Initialize your model here
    
    def load(self) -> None:
        """Load your model"""
        if self.is_loaded:
            return
        # Load your model (e.g., from file, download, etc.)
        self.is_loaded = True
    
    def predict(self, image: Image.Image, params: Dict[str, Any], 
                layout_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run inference"""
        if not self.is_loaded:
            self.load()
        
        # Your inference logic here
        boxes = []  # Your detected layout boxes
        
        return {"boxes": boxes}
    
    def unload(self) -> None:
        """Unload model to free memory"""
        self.is_loaded = False
```

### Step 2: Add Configuration

Add your model to the appropriate configuration file:

```json
// models/configs/layout_models.json
{
  "my_layout_model": {
    "type": "layout",
    "class_path": "models.layout.my_layout_model.MyLayoutModel",
    "config": {
      "description": "My custom layout detection model",
      "version": "1.0",
      "supports_tables": true,
      "custom_param": "value"
    }
  }
}
```

### Step 3: Use Your Model

Your model is now automatically available through the API:

```bash
# List available models
curl http://localhost:8000/models/

# Use your model
curl -X POST http://localhost:8000/infer/file \
  -F "file=@image.png" \
  -F "layout_model=my_layout_model"
```

## Configuration Files

### Layout Models (`models/configs/layout_models.json`)
```json
{
  "model_name": {
    "type": "layout",
    "class_path": "module.path.to.ModelClass",
    "config": {
      "description": "Model description",
      "version": "1.0",
      "supports_tables": true,
      "supports_figures": true,
      "custom_parameters": "values"
    }
  }
}
```

### OCR Models (`models/configs/ocr_models.json`)
```json
{
  "model_name": {
    "type": "ocr",
    "class_path": "module.path.to.ModelClass",
    "config": {
      "description": "Model description",
      "version": "1.0",
      "language": "eng",
      "supports_multiple_languages": true,
      "supports_layout_guidance": true
    }
  }
}
```

## API Endpoints

### Model Management
- `GET /models/` - List available models
- `GET /plugins/` - List all model plugins
- `GET /plugins/status` - Get model status (loaded/unloaded)
- `POST /plugins/reload` - Reload all model configurations
- `POST /plugins/{model_type}/{model_name}/load` - Load specific model
- `POST /plugins/{model_type}/{model_name}/unload` - Unload specific model
- `POST /plugins/unload-all` - Unload all models

### Inference
- `POST /infer/` - Base64 image inference
- `POST /infer/file` - File upload with JSON response
- `POST /infer/file-with-files` - File upload with ZIP download
- `POST /infer/file-json` - File upload with JSON only

## Model Interface Requirements

### Layout Models
Must implement:
- `load()` - Load the model
- `predict(image, params, layout_output=None)` - Return `{"boxes": [...]}`
- `unload()` - Free model resources

### OCR Models
Must implement:
- `load()` - Load the model
- `predict(image, params, layout_output=None)` - Return `{"tokens": [...]}`
- `unload()` - Free model resources

## Example Model Implementations

See the example models:
- `models/layout/example_layout_model.py` - Example layout model
- `models/ocr/example_ocr_model.py` - Example OCR model

## Running the Plugin System

### Start the Plugin-based Server
```bash
uvicorn backend.app_plugin:app --host 0.0.0.0 --port 8002 --reload
```

### Available Apps
- `backend.app:app` - Original monolithic app (port 8000)
- `backend.app_new:app` - MVC structured app (port 8001)
- `backend.app_plugin:app` - Plugin-based app (port 8002)

## Benefits

1. **Easy Model Addition**: Just create a class and add to config
2. **Dynamic Loading**: Models loaded/unloaded as needed
3. **Configuration Management**: JSON-based model configuration
4. **Memory Management**: Models can be unloaded to free memory
5. **Type Safety**: Abstract base classes ensure consistent interfaces
6. **Hot Reloading**: Model configurations can be reloaded without restart
7. **Extensibility**: Easy to add new model types beyond layout/OCR

## Best Practices

1. **Always implement `unload()`** to free GPU/memory resources
2. **Use configuration parameters** for model-specific settings
3. **Handle errors gracefully** in `load()` and `predict()` methods
4. **Validate input parameters** in your model classes
5. **Use descriptive model names** and descriptions
6. **Test your models** with the example endpoints before production use
