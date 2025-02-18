from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor
from tsml_eval.experiments import experiments, get_regressor_by_name, run_regression_experiment
from tsml_eval.evaluation.storage import load_regressor_results
from tsml.base import BaseTimeSeriesEstimator
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import roc_auc_score

from models.random_feature_representation_boosting import HydraBoost
from load_datasets import get_aeon_dataset

#ranked approximately by memory size (N_train * D * T) of each dataset 
TSER_datasets = [
    'BIDMC32HR',
    'BIDMC32SpO2',
    'PPGDalia',
    'BIDMC32RR',
    'NewsTitleSentiment',
    'NewsHeadlineSentiment',
    'DailyTemperatureLatitude',
    'LiveFuelMoistureContent',
    'IEEEPPG',
    'VentilatorPressure',
    'AustraliaRainfall',
    'BenzeneConcentration',
    'PhosphorusConcentration',
    'PotassiumConcentration',
    'MagnesiumConcentration',
    'ElectricMotorTemperature',
    'HouseholdPowerConsumption2',
    'HouseholdPowerConsumption1',
    'BeijingPM10Quality',
    'BeijingPM25Quality',
    'MadridPM10Quality',
    'GasSensorArrayEthanol',
    'GasSensorArrayAcetone',
    'CopperConcentration',
    'AluminiumConcentration',
    'BoronConcentration',
    'ZincConcentration',
    'CalciumConcentration',
    'SulphurConcentration',
    'IronConcentration',
    'ManganeseConcentration',
    'SodiumConcentration',
    'PrecipitationAndalusia',
    'ElectricityPredictor',
    'SolarRadiationAndalusia',
    'AppliancesEnergy',
    'ChilledWaterPredictor',
    'LPGasMonitoringHomeActivity',
    'MethaneMonitoringHomeActivity',
    'TetuanEnergyConsumption',
    'HotwaterPredictor',
    'BeijingIntAirportPM25Quality',
    'BarCrawl6min',
    'SteamPredictor',
    'FloodModeling1',
    'FloodModeling2',
    'FloodModeling3',
    'WindTurbinePower',
    'MetroInterstateTrafficVolume',
    'WaveDataTension',
    'AcousticContaminationMadrid',
    'OccupancyDetectionLight',
    'DhakaHourlyAirQuality',
    'SierraNevadaMountainsSnow',
    'ParkingBirmingham',
    'Covid19Andalusia',
    'EthereumSentiment',
    'Covid3Month',
    'BitcoinSentiment',
    'BinanceCoinSentiment',
    'DailyOilGasPrices',
    'CardanoSentiment',
    'NaturalGasPricesSentiment',
]

##########################################################
### SKLearn Wrapper for gridsearch, and class for tsml ###
##########################################################

class SKLearnWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, modelClass=None, **model_params,):
        self.modelClass = modelClass
        self.model_params = model_params
        self.seed = None
        self.model = None
        
        
    def set_params(self, **params):
        self.modelClass = params.pop('modelClass', self.modelClass)
        self.seed = params.pop('seed', self.seed)
        self.model_params.update(params)
        return self


    def get_params(self, deep=True):
        params = {'modelClass': self.modelClass}
        params.update(self.model_params)
        return params
    
    
    def fit(self, X, y):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        self.model = self.modelClass(**self.model_params)
        self.model.fit(X, y)
        # #classes, either label for binary or one-hot for multiclass
        # if len(y.size()) == 1 or y.size(1) == 1:
        #     self.classes_ = np.unique(y.detach().cpu().numpy())
        # else:
        #     self.classes_ = np.unique(y.argmax(axis=1).detach().cpu().numpy())
        return self


    def predict(self, X):
        return self.model(X).squeeze()#.detach().cpu().squeeze().numpy()
        # #binary classification
        # if len(self.classes_) == 2:
        #     proba_1 = torch.sigmoid(self.model(X))
        #     return (proba_1 > 0.5).detach().cpu().numpy()
        # else:
        #     #multiclass
        #     return torch.argmax(self.model(X), dim=1).detach().cpu().numpy()
    
    # def predict_proba(self, X):
    #     #binary classification
    #     if len(self.classes_) == 2:
    #         proba_1 = torch.nn.functional.sigmoid(self.model(X))
    #         return torch.cat((1 - proba_1, proba_1), dim=1).detach().cpu().numpy()
    #     else:
    #         #multiclass
    #         logits = self.model(X)
    #         proba = torch.nn.functional.softmax(logits, dim=1)
    #         return proba.detach().cpu().numpy()
    
    # def decision_function(self, X):
    #     logits = self.model(X)
    #     return logits.detach().cpu().numpy()


    
    # def score(self, X, y):
    #     logits = self.model(X)
    #     if y.size(1) == 1:
    #         y_true = y.detach().cpu().numpy()
    #         y_score = logits.detach().cpu().numpy()
    #         auc = roc_auc_score(y_true, y_score)
    #         return auc
    #     else:
    #         pred = torch.argmax(logits, dim=1)
    #         y = torch.argmax(y, dim=1)
    #         acc = (pred == y).float().mean()
    #         return acc.detach().cpu().item()
    
    
    
class TSMLGridSearchWrapper(RegressorMixin, BaseTimeSeriesEstimator):
    
    def __init__(self,
                 holdour_or_kfold: Literal["holdout", "kfold"] = "kfold",
                 kfolds: Optional[int] = 5,
                 holdout_percentage: Optional[float] = 0.2,
                 seed: Optional[int] = None,
                 modelClass=None, 
                 model_param_grid: Dict[str, List[Any]] = {},
                 device: str = "cpu",
                 verbose:int = 1,  # 0 1 2
        ):
        self.holdour_or_kfold = holdour_or_kfold
        self.kfolds = kfolds
        self.holdout_percentage = holdout_percentage
        self.seed = seed
        self.modelClass = modelClass
        self.model_param_grid = model_param_grid
        self.device = device
        self.verbose = verbose
        super(TSMLGridSearchWrapper, self).__init__()
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """Fit the estimator to training data, with gridsearch hyperparameter optimization
        on holdout or kfold cross-validation.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The target labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        # TODO regression only
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
        y = y.unsqueeze(1)
        self.X_mean = X.mean()
        self.X_std = X.std()
        self.y_mean = y.mean()
        self.y_std = y.std()
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
        
        # Configure cross validation
        if self.holdour_or_kfold == "kfold":
            cv = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)
        else:  # holdout
            cv = ShuffleSplit(n_splits=1, test_size=self.holdout_percentage, random_state=self.seed)
                
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=SKLearnWrapper(modelClass=self.modelClass),
            param_grid={**self.model_param_grid, "seed": [self.seed]},
            cv=cv,
            scoring="neg_mean_squared_error", # TODO regression only???
            verbose=self.verbose
        )
        grid_search.fit(X, y)

        # Store best model
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        print("self.best_params", self.best_params)
        return self
        
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted target labels.
        """
        X = torch.from_numpy(X).float()
        X = (X - self.X_mean) / self.X_std
        pred = self.best_model.predict(X) #TODO regression only?
        pred = pred * self.y_std + self.y_mean
        return pred.squeeze().detach().cpu().numpy()
        

    def _more_tags(self) -> dict:
        return {
            "X_types": ["3darray"],
            "equal_length_only": True,
            "allow_nan": False,
        }
        
        
    def get_params(self):
        """Use for saving model configuration in tsml"""
        if hasattr(self, 'best_params'):
            return {
            "seed": self.seed,
            **self.best_params
            }
        else:
            return {}
        
        
        
        
def test_regressor(
        regressor_name, # = "HydraBoost",
        regressor, #= TSMLWrapperHydraBoost(),
        dataset_name, #= "HouseholdPowerConsumption1",
        TSER_data_dir,
        results_dir,
        resample_id=0,
    ):
    #get HouseholdPowerConsumption1 dataset
    X_train, y_train, X_test, y_test = get_aeon_dataset(dataset_name, TSER_data_dir, "regression")

    #run regression experiment
    run_regression_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        regressor,
        regressor_name = regressor_name,
        results_path = results_dir,
        dataset_name = dataset_name,
        resample_id=resample_id,
    )
    rr = load_regressor_results(
        results_dir / regressor_name / "Predictions" / dataset_name / f"testResample{resample_id}.csv"
    )
    print(rr.mean_squared_error, "mse")
    print(rr.root_mean_squared_error, "rmse")
    print(rr.mean_absolute_percentage_error, "mape")
    print(rr.r2_score, "r2")
    print(rr.fit_time, "fit time")
        


######################################################  |
#####  command line argument to run experiments  #####  |
######################################################  V



def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with different models and datasets.")
    # parser.add_argument(
    #     "--models", 
    #     nargs='+', 
    #     type=str, 
    #     default=["HydraBoost"], 
    #     help="List of model names to run."
    # )
    parser.add_argument(
        "--dataset_indices", 
        nargs='+', 
        type=int, 
        default=[i for i in range(len(TSER_datasets))], 
        help="List of datasets to run."
    )
    parser.add_argument(
        "--resample_ids", 
        nargs='+', 
        type=int, 
        default = [0],
        #default=[i for i in range(30)], 
        help="What tsml resampling to use. 0 is default split. Ranges up to 29 inclusive."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="C:\\Users\\nz423\\Code\\exploring-hydra-boosting\\results",
        help="Directory where the results will be saved to file."
    )
    parser.add_argument(
        "--TSER_dir",
        type=str,
        default="C:\\Users\\nz423\\Data\\TSER",
        help="Location of the datasets."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to run the experiments on."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for all randomness."
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    for ID in args.dataset_indices:
        for resample_id in args.resample_ids:
            test_regressor(
                regressor_name = "HydraBoostGridSearchHoldout",
                regressor = TSMLGridSearchWrapper(
                            "holdout",
                            seed=args.seed,
                            device=args.device,
                            modelClass=HydraBoost,
                            model_param_grid={
                                "n_layers": [0, 1, 3, 6],              # [0,1,3,6,10] ?
                                "init_n_kernels": [8],
                                "init_n_groups": [64],
                                "n_kernels": [8],
                                "n_groups": [64],
                                "max_num_channels": [3],
                                "hydra_batch_size": [1024],
                                "l2_reg": [0.01, 0.1, 1, 10, 100],                # [0.01, 0.1, 1, 10] ?
                                "l2_ghat": [0.01, 0.1, 1, 10],          # [0.01, 0.1, 1, 10] ?
                                "boost_lr": [0.5],
                            },
                            ),
                dataset_name = TSER_datasets[ID],
                TSER_data_dir = Path(args.TSER_dir),
                results_dir = Path(args.results_dir),
                resample_id=resample_id,
            )