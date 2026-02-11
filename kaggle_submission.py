"""
Kaggle Submission Generator for Zillow Prize
Creates submission.csv with predictions for each month in format:
ParcelId,201610,201611,201612,201710,201711,201712

This script loads trained models and generates predictions for all properties
"""

import pandas as pd
import numpy as np
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KaggleSubmissionGenerator:
    """Generate Kaggle submission CSV with monthly predictions"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.predictions_by_month = {}
        self.parcel_ids = None
        
    def load_test_data(self):
        """Load test properties (no labels)"""
        logger.info("Loading test data...")
        
        # Load 2016 and 2017 property data
        # Try data_dir first, then project root
        def _try_load(fname):
            p = Path(self.data_dir) / fname
            if not p.exists():
                p = Path(fname)
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    logger.info(f"Loaded {p}: {df.shape}")
                    return df
                except Exception as e:
                    logger.error(f"Failed to read {p}: {e}")
            return None

        properties_2016 = _try_load('properties_2016.csv')
        properties_2017 = _try_load('properties_2017.csv')
        
        # Combine
        if properties_2016 is not None and properties_2017 is not None:
            properties = pd.concat([properties_2016, properties_2017], ignore_index=True).drop_duplicates(subset=['parcelid'])
        elif properties_2016 is not None:
            properties = properties_2016
        else:
            properties = properties_2017
        
        logger.info(f"Total unique properties: {len(properties)}")
        self.parcel_ids = properties['parcelid'].values
        
        return properties
    
    def load_sample_submission(self):
        """Load sample submission format"""
        logger.info("Loading sample submission format...")
        
        try:
            sample = pd.read_csv(self.data_dir / 'sample_submission.csv')
            logger.info(f"Sample submission shape: {sample.shape}")
            logger.info(f"Columns: {sample.columns.tolist()}")
            return sample
        except Exception as e:
            logger.error(f"Could not load sample submission: {e}")
            # Create default format
            months = ['201610', '201611', '201612', '201710', '201711', '201712']
            sample = pd.DataFrame({'ParcelId': self.parcel_ids})
            for month in months:
                sample[month] = 0.0
            return sample
    
    def prepare_features(self, properties):
        """Prepare features for prediction (same as training)"""
        logger.info("Preparing features...")
        
        from src.data_preparation import DataPreparation
        
        # Apply same preprocessing as training
        prep = DataPreparation(data_dir=self.data_dir)
        
        # Fill missing values
        numeric_cols = properties.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            properties[col] = properties[col].fillna(properties[col].median())
        
        # Feature engineering
        if 'yearbuilt' in properties.columns:
            properties['property_age'] = 2017 - properties['yearbuilt']
        
        if 'calculatedfinishedsquarefeet' in properties.columns and 'taxvaluedollarcnt' in properties.columns:
            properties['price_per_sqft'] = properties['taxvaluedollarcnt'] / (properties['calculatedfinishedsquarefeet'] + 1)
        
        if 'bedroomcnt' in properties.columns and 'bathroomcnt' in properties.columns:
            properties['bed_bath_ratio'] = properties['bedroomcnt'] / (properties['bathroomcnt'] + 1)
            properties['total_rooms'] = properties['bedroomcnt'] + properties['bathroomcnt']
        
        # Select numeric features
        numeric_cols = properties.select_dtypes(include=[np.number]).columns.tolist()
        X = properties[numeric_cols]
        
        logger.info(f"Features prepared: {X.shape}")
        return X
    
    def generate_predictions_lightgbm(self, X, sample):
        """Generate predictions using LightGBM (best performing model)"""
        logger.info("\nGenerating predictions with LightGBM...")
        
        try:
            # Train simple LightGBM on all training data for submission
            from src.data_preparation import DataPreparation
            
            # Load training data
            prep = DataPreparation(data_dir=self.data_dir)
            data_2016, _, train_2016 = prep.prepare_data('properties_2016.csv', 'train_2016_v2.csv')
            data_2017, _, train_2017 = prep.prepare_data('properties_2017.csv', 'train_2017.csv')
            
            # Combine training data
            train_data = pd.concat([data_2016, data_2017], ignore_index=True)
            
            # Prepare features
            numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
            features = [c for c in numeric_cols if c not in ['logerror', 'parcelid']]
            
            X_train = train_data[features].fillna(train_data[features].median())
            y_train = train_data['logerror']
            
            # Train LightGBM
            train_dataset = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'device': 'gpu',
                'gpu_device_id': 0,
                'max_depth': 7,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
            }
            
            model = lgb.train(params, train_dataset, num_boost_round=500)
            
            # Predict on test set
            X_test = X.copy()
            predictions = model.predict(X_test)
            
            logger.info(f"✓ LightGBM predictions generated: {len(predictions)}")
            return predictions
            
        except Exception as e:
            logger.error(f"LightGBM prediction failed: {e}")
            # Return default predictions (median logerror from training)
            return np.full(len(X), 0.005)
    
    def generate_predictions_rf(self, X):
        """Train a RandomForest on the combined train data and predict for X.

        This mirrors the settings used in `assign.py` (n_estimators=200, random_state=42).
        """
        logger.info("Generating predictions with RandomForest (train on full training set)...")
        try:
            # Load properties (for merging) and train files
            props = []
            for fname in ["properties_2016.csv", "properties_2017.csv"]:
                p = Path(self.data_dir) / fname
                if p.exists():
                    props.append(pd.read_csv(p))
            properties_all = pd.concat(props, ignore_index=True).drop_duplicates(subset=["parcelid"]) if props else pd.DataFrame()

            trains = []
            for tname in ["train_2016_v2.csv", "train_2017.csv"]:
                t = Path(self.data_dir) / tname
                if not t.exists():
                    t = Path(tname)
                if t.exists():
                    trains.append(pd.read_csv(t, parse_dates=["transactiondate"]))
            if not trains:
                raise FileNotFoundError("No train files found in data/ to train RandomForest fallback.")

            train_df = pd.concat(trains, ignore_index=True)
            # merge with properties to get features
            if not properties_all.empty:
                train_df = train_df.merge(properties_all, how="left", on="parcelid")

            # feature engineering similar to prepare_features
            if "transactiondate" in train_df.columns:
                train_df["transaction_month"] = pd.to_datetime(train_df["transactiondate"]).dt.month
                train_df["transaction_year"] = pd.to_datetime(train_df["transactiondate"]).dt.year
            if "yearbuilt" in train_df.columns and "transaction_year" in train_df.columns:
                train_df["age"] = (train_df["transaction_year"] - train_df["yearbuilt"]).clip(lower=0)

            sqft_cols = ["calculatedfinishedsquarefeet", "finishedsquarefeet12", "finishedsquarefeet13"]
            if any(c in train_df.columns for c in sqft_cols):
                train_df["sqft"] = train_df[sqft_cols].bfill(axis=1).iloc[:, 0]

            if "taxvaluedollarcnt" in train_df.columns and "calculatedfinishedsquarefeet" in train_df.columns:
                train_df["price_per_sqft"] = train_df["taxvaluedollarcnt"] / (train_df["calculatedfinishedsquarefeet"] + 1)
            if "bedroomcnt" in train_df.columns and "bathroomcnt" in train_df.columns:
                train_df["bed_bath_ratio"] = train_df["bedroomcnt"] / (train_df["bathroomcnt"] + 1)
                train_df["total_rooms"] = train_df["bedroomcnt"] + train_df["bathroomcnt"]

            # Keep numeric features only
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            # exclude ids/target
            features = [c for c in numeric_cols if c not in ["logerror", "parcelid"]]
            X_train = train_df[features].copy()
            y_train = train_df["logerror"].copy()

            # impute
            X_train = X_train.fillna(X_train.median())

            # Ensure X has the same columns as provided X (prepare_features previously)
            X_test = X.copy()
            # If X contains derived cols not in train features, add them; otherwise align
            for col in features:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[features].fillna(X_train.median())

            # Train RF with same settings as assign
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            # save model
            joblib.dump(rf, Path("models") / "kaggle_rf_model.joblib")
            logger.info(f"✓ RandomForest trained and predictions generated: {len(preds)}")
            return preds
        except Exception as e:
            logger.error(f"RandomForest prediction failed: {e}")
            return np.full(len(X), 0.005)
    def generate_predictions_xgboost(self, X):
        """Generate predictions using XGBoost"""
        logger.info("Generating predictions with XGBoost...")
        
        try:
            from src.data_preparation import DataPreparation
            
            # Load training data
            prep = DataPreparation(data_dir=self.data_dir)
            data_2016, _, _ = prep.prepare_data('properties_2016.csv', 'train_2016_v2.csv')
            data_2017, _, _ = prep.prepare_data('properties_2017.csv', 'train_2017.csv')
            
            # Combine
            train_data = pd.concat([data_2016, data_2017], ignore_index=True)
            
            # Prepare features
            numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
            features = [c for c in numeric_cols if c not in ['logerror', 'parcelid']]
            
            X_train = train_data[features].fillna(train_data[features].median())
            y_train = train_data['logerror']
            
            # Train XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            params = {
                'objective': 'reg:squarederror',
                'metric': 'mae',
                'tree_method': 'hist',
                'gpu_id': 0,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
            
            model = xgb.train(params, dtrain, num_boost_round=500)
            
            # Predict
            X_test = X.copy()
            dtest = xgb.DMatrix(X_test)
            predictions = model.predict(dtest)
            
            logger.info(f"✓ XGBoost predictions generated: {len(predictions)}")
            return predictions
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return np.full(len(X), 0.005)
    
    def ensemble_predictions(self, lightgbm_pred, xgboost_pred):
        """Blend predictions from multiple models (LightGBM best performing)"""
        logger.info("Blending predictions...")
        
        # LightGBM had best MAE (0.0347), so weight it more
        # Weights: LightGBM 60%, XGBoost 40%
        blended = 0.6 * lightgbm_pred + 0.4 * xgboost_pred
        
        logger.info(f"✓ Predictions blended (LightGBM 60%, XGBoost 40%)")
        return blended
    
    def create_submission_csv(self, predictions, sample):
        """Create submission CSV with predictions for each month"""
        logger.info("Creating submission CSV...")
        
        # Months in Kaggle format
        months = ['201610', '201611', '201612', '201710', '201711', '201712']
        
        # Create submission dataframe
        submission = pd.DataFrame()
        submission['ParcelId'] = sample['ParcelId'].values
        
        # For each month, use same predictions (model predicts logerror for any month)
        for month in months:
            submission[month] = predictions
        
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Columns: {submission.columns.tolist()}")
        logger.info(f"\nFirst 5 rows:")
        logger.info(submission.head())
        
        return submission
    
    def save_submission(self, submission, output_path='outputs/submission.csv'):
        """Save submission to CSV"""
        submission.to_csv(output_path, index=False)
        logger.info(f"\n✓ Submission saved to {output_path}")
        logger.info(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
        
        return output_path
    
    def generate_full_submission(self):
        """Complete pipeline to generate Kaggle submission"""
        logger.info("="*80)
        logger.info("KAGGLE SUBMISSION GENERATOR")
        logger.info("="*80)
        
        # Load data
        properties = self.load_test_data()
        sample = self.load_sample_submission()
        
        # Generate predictions using RandomForest (best model)
        # pass raw properties dataframe; generate_predictions_rf will prepare features
        final_pred = self.generate_predictions_rf(properties)

        # Create submission
        submission = self.create_submission_csv(final_pred, sample)

        # Save
        output_file = self.save_submission(submission)
        
        logger.info("\n" + "="*80)
        logger.info("✓ SUBMISSION READY!")
        logger.info("="*80)
        logger.info(f"Output file: {output_file}")
        logger.info(f"Columns: ParcelId, 201610, 201611, 201612, 201710, 201711, 201712")
        logger.info(f"Ready to upload to Kaggle!")
        
        return submission


if __name__ == "__main__":
    generator = KaggleSubmissionGenerator(data_dir='data')
    submission = generator.generate_full_submission()