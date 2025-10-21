# Backend Architecture

## Overview
The backend has been restructured into a clean MVC (Model-View-Controller) architecture with proper separation of concerns.

## Folder Structure

```
backend/
├── controllers/          # API Controllers (handles HTTP requests/responses)
│   ├── __init__.py
│   ├── health_controller.py
│   ├── model_controller.py
│   └── inference_controller.py
├── routes/              # Route definitions (URL routing)
│   ├── __init__.py
│   ├── health_routes.py
│   ├── model_routes.py
│   └── inference_routes.py
├── services/            # Business logic layer
│   ├── __init__.py
│   ├── inference_service.py
│   ├── file_service.py
│   └── model_service.py
├── helpers/             # Helper classes for common operations
│   ├── __init__.py
│   └── image_helper.py
├── utils/               # Utility functions
│   ├── visualize.py
│   ├── response_utils.py
│   ├── validation_utils.py
│   └── config.py
├── models/              # Model adapters and registry
│   ├── registry.py
│   ├── layout/
│   └── ocr/
├── app.py              # Original monolithic app
├── app_new.py          # New structured app
└── requirements.txt
```

## Architecture Layers

### 1. Controllers (`controllers/`)
- **Purpose**: Handle HTTP requests and responses
- **Responsibilities**:
  - Parse request data
  - Call appropriate services
  - Format responses
  - Handle HTTP-specific concerns

**Files:**
- `health_controller.py`: Health check operations
- `model_controller.py`: Model listing and information
- `inference_controller.py`: All inference operations

### 2. Routes (`routes/`)
- **Purpose**: Define API endpoints and URL routing
- **Responsibilities**:
  - Define endpoint paths
  - Handle HTTP methods (GET, POST, etc.)
  - Apply middleware and decorators
  - Route requests to controllers

**Files:**
- `health_routes.py`: `/health/` endpoints
- `model_routes.py`: `/models/` endpoints  
- `inference_routes.py`: `/infer/` endpoints

### 3. Services (`services/`)
- **Purpose**: Business logic and core functionality
- **Responsibilities**:
  - Implement business rules
  - Coordinate between different components
  - Handle data processing
  - Manage model operations

**Files:**
- `inference_service.py`: Model inference operations
- `file_service.py`: File processing and validation
- `model_service.py`: Model registry management

### 4. Helpers (`helpers/`)
- **Purpose**: Reusable helper classes for common operations
- **Responsibilities**:
  - Provide utility methods for specific domains
  - Encapsulate common patterns
  - Simplify complex operations

**Files:**
- `image_helper.py`: Image processing utilities

### 5. Utils (`utils/`)
- **Purpose**: General utility functions
- **Responsibilities**:
  - Provide reusable functions
  - Handle configuration
  - Manage responses and validation

**Files:**
- `visualize.py`: Image visualization functions
- `response_utils.py`: Standardized API responses
- `validation_utils.py`: Input validation utilities
- `config.py`: Application configuration

## API Endpoints

### Health Endpoints
- `GET /health/` - Health check

### Model Endpoints
- `GET /models/` - List available models
- `GET /models/info` - Get detailed model information

### Inference Endpoints
- `POST /infer/` - Base64 image inference
- `POST /infer/file` - File upload with JSON response
- `POST /infer/file-with-files` - File upload with ZIP download
- `POST /infer/file-json` - File upload with JSON only

## Benefits of New Structure

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Maintainability**: Code is organized and easy to find
3. **Testability**: Each component can be tested independently
4. **Scalability**: Easy to add new features without affecting existing code
5. **Reusability**: Services and helpers can be reused across controllers
6. **Configuration**: Centralized configuration management
7. **Error Handling**: Standardized error responses

## Migration

The original `app.py` is preserved for backward compatibility. The new structured app is in `app_new.py`.

To use the new structure:
```bash
uvicorn backend.app_new:app --host 0.0.0.0 --port 8001 --reload
```

To use the original structure:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
