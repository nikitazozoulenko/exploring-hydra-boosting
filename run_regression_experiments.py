from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import argparse
from pathlib import Path
import abc

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
import optuna

from models.random_feature_representation_boosting import HydraFeatureBoost
from models.label_space_boost import HydraLabelBoost, HydraLabelReuseBoost
from models.naive import NaiveMean
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
    
    
    def fit(self, X:Tensor, y:Tensor):
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


    def predict(self, X:Tensor):
        return self.model(X)
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


    
    def score(self, X:Tensor, y:Tensor):
        with torch.no_grad():
            y_pred = self.model(X)
            mse = torch.nn.functional.mse_loss(y_pred, y)
            return -mse.detach().cpu().item()  # Return negative MSE since sklearn maximizes scores TODO only regression for now
        # logits = self.model(X)
        # if y.size(1) == 1:
        #     y_true = y.detach().cpu().numpy()
        #     y_score = logits.detach().cpu().numpy()
        #     auc = roc_auc_score(y_true, y_score)
        #     return auc
        # else:
        #     pred = torch.argmax(logits, dim=1)
        #     y = torch.argmax(y, dim=1)
        #     acc = (pred == y).float().mean()
        #     return acc.detach().cpu().item()
    
    
class TSMLBaseWrapper(RegressorMixin, BaseTimeSeriesEstimator):
    
    def __init__(self,
                 holdour_or_kfold: Literal["holdout", "kfold"] = "kfold",
                 kfolds: Optional[int] = 5,
                 holdout_percentage: Optional[float] = 0.2,
                 seed: Optional[int] = None,
                 device: str = "cpu",
                 modelClass=None, 
                 **kwargs,
                 ):
        self.holdour_or_kfold = holdour_or_kfold
        self.kfolds = kfolds
        self.holdout_percentage = holdout_percentage
        self.seed = seed
        self.device = device
        self.modelClass = modelClass
        super(TSMLBaseWrapper, self).__init__()
    
    
    def _hyperopt_tune(self, X:Tensor, y:Tensor, cv:Union[KFold, ShuffleSplit]):
        raise NotImplementedError("Subclasses must implement _hyperopt_tune method.")
        
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """Fit the estimator to training data, using function TODO

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
        # TODO regression only currently
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
            
        self.best_model, self.best_params = self._hyperopt_tune(X, y, cv)
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
        X = torch.from_numpy(X).float().to(self.device)
        X = (X - self.X_mean) / self.X_std
        pred = self.best_model(X) #TODO regression only?
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
        


class TSMLGridSearchWrapper(TSMLBaseWrapper):
    def __init__(self,
                    holdour_or_kfold: Literal["holdout", "kfold"] = "kfold",
                    kfolds: Optional[int] = 5,
                    holdout_percentage: Optional[float] = 0.2,
                    seed: Optional[int] = None,
                    device: str = "cpu",
                    modelClass=None, 
                    model_param_grid: Dict[str, List[Any]] = {},
                    verbose:int = 2,  # 0 1 2
            ):
        self.model_param_grid = model_param_grid
        self.verbose = verbose
        super(TSMLGridSearchWrapper, self).__init__(
            holdour_or_kfold, kfolds, holdout_percentage, seed, device, modelClass, 
        )
        
        
    def _hyperopt_tune(self, X:Tensor, y:Tensor, cv:Union[KFold, ShuffleSplit]):
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=SKLearnWrapper(modelClass=self.modelClass),
            param_grid={**self.model_param_grid, "seed": [self.seed]},
            cv=cv,
            verbose=self.verbose
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_.model, grid_search.best_params_



############################################################
### Optuna Wrapper class for hyperparameter optimization ###
############################################################




def get_optuna_objective(
    trial,
    modelClass: Callable,
    get_optuna_params: Callable,
    X_train: Tensor,
    y_train: Tensor,
    cv: Union[KFold, ShuffleSplit],
    seed: int,
    regression_or_classification: Literal["regression", "classification"] = "regression",
    ):
    """The objective (pre-lambdaing) to be minimzed in Optuna's 'study.optimize(objective, n_trails)' function."""
    try:
        params = get_optuna_params(trial)
        scores = []
        
        # Use the provided cv parameter directly
        for train_idx, valid_idx in cv.split(X_train):
            X_inner_train, X_inner_valid = X_train[train_idx], X_train[valid_idx]
            y_inner_train, y_inner_valid = y_train[train_idx], y_train[valid_idx]

            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            model = modelClass(**params)
            model.fit(X_inner_train, y_inner_train)

            preds = model(X_inner_valid)
            if regression_or_classification == "classification":
                if y_inner_valid.shape[1] > 2:  # Multiclass classification 
                    preds = torch.argmax(preds, dim=1)
                    gt = torch.argmax(y_inner_valid, dim=1)
                    acc = (preds == gt).float().mean()
                    scores.append(-acc.item())  # score is being minimized in Optuna
                else:  # Binary classification
                    preds = torch.sigmoid(preds).round()
                    acc = (preds == y_inner_valid).float().mean()
                    scores.append(-acc.item())  # score is being minimized in Optuna
            else:
                rmse = torch.sqrt(nn.functional.mse_loss(y_inner_valid, preds))
                scores.append(rmse.item())

        return np.mean(scores)
    except (RuntimeError, ValueError, torch._C._LinAlgError) as e:
        print(f"Error encountered during training: {e}")
        return 10.0 #rmse random guessing is 1.0, and random guessing accuracy is -1/C
    


    
def get_optuna_hydrafeatureboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_layers": trial.suggest_int("n_layers", 0, 9),
        "l2_reg": trial.suggest_float("l2_reg", 0.1, 1000, log=True),
        "l2_ghat": trial.suggest_float("l2_ghat", 0.01, 100, log=True),
        "boost_lr": trial.suggest_float("boost_lr", 0.1, 1.0),
    }


def get_optuna_hydra_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [1]),
        "l2_reg": trial.suggest_float("l2_reg", 0.1, 1000, log=True),
    }
    
    
def get_optuna_hydralabelboost_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1, 10),
        "l2_reg": trial.suggest_float("l2_reg", 0.1, 1000, log=True),
        "boost_lr": trial.suggest_float("boost_lr", 0.1, 1.0),
    }
    
    
def get_optuna_naivemean_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {}


class TSMLOptunaWrapper(TSMLBaseWrapper):
    def __init__(self,
                 holdour_or_kfold: Literal["holdout", "kfold"] = "kfold",
                 kfolds: Optional[int] = 5,
                 holdout_percentage: Optional[float] = 0.2,
                 seed: Optional[int] = None,
                 modelClass = None,
                 optuna_param_func: Callable = None,
                 device: str = "cpu",
                 n_trials: int = 100,
        ):
        self.optuna_param_func = optuna_param_func
        self.n_trials = n_trials
        super(TSMLOptunaWrapper, self).__init__(
            holdour_or_kfold, kfolds, holdout_percentage, seed, device, modelClass,
        )
        
        
    def _hyperopt_tune(self, X:Tensor, y:Tensor, cv:Union[KFold, ShuffleSplit]):
        #hyperparameter tuning with Optuna
        sampler = optuna.samplers.TPESampler(seed=self.seed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction="minimize", sampler=sampler)
        objective = lambda trial: get_optuna_objective(
            trial, self.modelClass, self.optuna_param_func, 
            X, y, cv, self.seed, "regression"
            )
        study.optimize(objective, self.n_trials)

        #fit model with optimal hyperparams
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        model = self.modelClass(**study.best_params)
        model.fit(X, y)
        return model, study.best_params
        

########################################
## tsml wrapper
########################################

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
    parser.add_argument(
        "--models", 
        nargs='+', 
        type=str, 
        default=["HydraFeatureBoost", "HydraLabelBoost", "NaiveMean"], 
        help="List of model names to run."
    )
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
    parser.add_argument(
        "--optuna_or_gridsearch",
        type=str,
        default="optuna",
    )
    parser.add_argument(
        "--holdout_or_kfold",
        type=str,
        default="holdout",
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=100,
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    for model_name in args.models:
        #get model param grids for gridsearch and optuna ranges
        if "Hydra"==model_name:
            modelClass=HydraLabelBoost
            optuna_param_func = get_optuna_hydra_params
            param_grid={
                    "n_estimators": [1],
                    "l2_reg": [1000, 100, 10, 1, 0.1],
                }
        elif "HydraFeatureBoost" in model_name:
            modelClass=HydraFeatureBoost
            optuna_param_func = get_optuna_hydrafeatureboost_params
            param_grid={
                    "n_layers": [0, 1, 3, 6, 10],
                    "l2_reg": [1000, 100, 10, 1, 0.1],
                    "l2_ghat": [0.01],
                    "boost_lr": [0.5],
                },
        elif ("HydraLabelBoost" in model_name) or ("HydraLabelReuseBoost" in model_name):
            modelClass=HydraLabelBoost if "HydraLabelBoost" in model_name else HydraLabelReuseBoost
            optuna_param_func = get_optuna_hydralabelboost_params
            param_grid={
                    "n_layers": [0, 1, 3, 6, 10],
                    "l2_reg": [1000, 100, 10, 1, 0.1],
                    "boost_lr": [0.5],
                }
        elif "NaiveMean" in model_name:
            modelClass=NaiveMean
            optuna_param_func = get_optuna_naivemean_params
            param_grid={}
        else:
            raise ValueError(f"Invalid model name given: {model_name}")
        
        
        # select the correct wrapper for optuna or gridsearch
        if args.optuna_or_gridsearch == "optuna":
            regressor = TSMLOptunaWrapper(
                holdour_or_kfold=args.holdout_or_kfold,
                kfolds=5,
                holdout_percentage=0.2,
                seed=args.seed,
                device=args.device,
                modelClass=modelClass,
                optuna_param_func=optuna_param_func,
                n_trials=args.n_optuna_trials,
            )
        else:
            regressor = TSMLGridSearchWrapper(
                holdour_or_kfold=args.holdout_or_kfold,
                kfolds=5,
                holdout_percentage=0.2,
                seed=args.seed,
                device=args.device,
                modelClass=modelClass,
                model_param_grid=param_grid,
            )
        
        #modify name based on optuna or gridsearch
        if args.optuna_or_gridsearch == "optuna":
            model_name = model_name + "_O"
        else:
            model_name = model_name + "_GS" 
        if args.holdout_or_kfold == "holdout":
            model_name = model_name + "H"
        else:
            model_name = model_name + "CV"
    
        #run experiments
        for dataset_idx in args.dataset_indices:
            for resample_id in args.resample_ids:
                print(args.TSER_dir)
                print(args.results_dir)
                test_regressor(
                    regressor_name = model_name,
                    regressor = regressor,
                    dataset_name = TSER_datasets[dataset_idx],
                    TSER_data_dir = Path(args.TSER_dir),
                    results_dir = Path(args.results_dir),
                    resample_id=resample_id,
                )