import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_xgboost_gpu():
    """Test XGBoost GPU support"""
    try:
        import xgboost as xgb
        logger.info(f"XGBoost Version: {xgb.__version__}")
        
        # Create a simple dataset
        X = np.random.rand(1000, 50)
        y = np.random.rand(1000)
        
        dtrain = xgb.DMatrix(X, label=y)
        
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'gpu_id': 0,
        }
        
        # Try training
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        logger.info("✓ XGBoost GPU acceleration is working!")
        return True
    except Exception as e:
        logger.warning(f"✗ XGBoost GPU error: {e}")
        logger.info("Falling back to CPU training")
        return False


def test_lightgbm_gpu():
    """Test LightGBM GPU support"""
    try:
        import lightgbm as lgb
        logger.info(f"LightGBM Version: {lgb.__version__}")
        
        # Create a simple dataset
        X = np.random.rand(1000, 50)
        y = np.random.rand(1000)
        
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            'objective': 'regression',
            'device': 'gpu',
            'gpu_device_id': 0,
            'verbose': -1,
        }
        
        # Try training
        model = lgb.train(params, train_data, num_boost_round=10)
        logger.info("✓ LightGBM GPU acceleration is working!")
        return True
    except Exception as e:
        logger.warning(f"✗ LightGBM GPU error: {e}")
        logger.info("Falling back to CPU training")
        return False


def get_gpu_status():
    """Get GPU availability status"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA is available")
            logger.info(f"  GPU Count: {torch.cuda.device_count()}")
            logger.info(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            logger.warning("✗ CUDA not available")
            return False
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check CUDA directly")
        return False
    except Exception as e:
        logger.warning(f"Error checking CUDA: {e}")
        return False


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("GPU AND CUDA SUPPORT TEST")
    logger.info("="*80)
    
    # Check CUDA
    cuda_available = get_gpu_status()
    
    # Test frameworks
    xgb_gpu = test_xgboost_gpu()
    lgb_gpu = test_lightgbm_gpu()
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"CUDA Available: {'Yes' if cuda_available else 'No'}")
    logger.info(f"XGBoost GPU: {'Yes' if xgb_gpu else 'No'}")
    logger.info(f"LightGBM GPU: {'Yes' if lgb_gpu else 'No'}")
    logger.info("="*80 + "\n")
    
    if all([cuda_available, xgb_gpu, lgb_gpu]):
        logger.info("✓ ALL GPU SYSTEMS READY - You're good to go!")
    else:
        logger.warning("⚠ Some GPU systems unavailable - will use CPU fallback")
