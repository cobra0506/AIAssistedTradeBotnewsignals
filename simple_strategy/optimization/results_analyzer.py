import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from optuna import visualization
import logging

class ResultsAnalyzer:
    """Analyze and visualize optimization results"""
    
    def __init__(self, study=None):
        """
        Initialize ResultsAnalyzer
        
        Args:
            study: Optuna study object (optional)
        """
        self.study = study
        self.logger = logging.getLogger(__name__)
    
    def set_study(self, study):
        """Set the Optuna study for analysis"""
        self.study = study
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters from the study"""
        if self.study is None:
            return None
        return self.study.best_params
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score from the study"""
        if self.study is None:
            return None
        return self.study.best_value
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get optimization trials as DataFrame"""
        if self.study is None:
            return pd.DataFrame()
        return self.study.trials_dataframe()
    
    def get_parameter_importance(self) -> Optional[Dict[str, float]]:
        """Get parameter importance scores"""
        if self.study is None:
            return None
        
        try:
            importances = optuna.importance.get_param_importances(self.study)
            return importances
        except Exception as e:
            self.logger.warning(f"Could not calculate parameter importance: {e}")
            return None
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history with scores"""
        if self.study is None:
            return pd.DataFrame()
        
        trials_df = self.study.trials_dataframe()
        if 'value' in trials_df.columns:
            return trials_df[['number', 'value', 'datetime_start', 'datetime_complete']].copy()
        return pd.DataFrame()
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history"""
        if self.study is None:
            self.logger.error("No study set for plotting")
            return
        
        try:
            fig = visualization.plot_optimization_history(self.study)
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
        except Exception as e:
            self.logger.error(f"Could not plot optimization history: {e}")
    
    def plot_parameter_importance(self, save_path: str = None):
        """Plot parameter importance"""
        if self.study is None:
            self.logger.error("No study set for plotting")
            return
        
        try:
            fig = visualization.plot_param_importances(self.study)
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
        except Exception as e:
            self.logger.error(f"Could not plot parameter importance: {e}")
    
    def plot_parallel_coordinate(self, save_path: str = None):
        """Plot parallel coordinate plot"""
        if self.study is None:
            self.logger.error("No study set for plotting")
            return
        
        try:
            fig = visualization.plot_parallel_coordinate(self.study)
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
        except Exception as e:
            self.logger.error(f"Could not plot parallel coordinate: {e}")
    
    def plot_contour(self, params: List[str] = None, save_path: str = None):
        """Plot contour plot"""
        if self.study is None:
            self.logger.error("No study set for plotting")
            return
        
        try:
            fig = visualization.plot_contour(self.study, params=params)
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
        except Exception as e:
            self.logger.error(f"Could not plot contour: {e}")
    
    def generate_summary_report(self, save_path: str = None) -> str:
        """Generate a summary report of the optimization"""
        if self.study is None:
            return "No study available for analysis"
        
        report = []
        report.append("=== OPTIMIZATION SUMMARY REPORT ===\n")
        
        # Basic stats
        report.append(f"Study Name: {self.study.study_name}")
        report.append(f"Direction: {self.study.direction}")
        report.append(f"Total Trials: {len(self.study.trials)}")
        report.append(f"Best Score: {self.study.best_value:.4f}")
        report.append("")
        
        # Best parameters
        report.append("Best Parameters:")
        for param, value in self.study.best_params.items():
            report.append(f"  {param}: {value}")
        report.append("")
        
        # Parameter importance
        importance = self.get_parameter_importance()
        if importance:
            report.append("Parameter Importance:")
            for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {param}: {score:.4f}")
            report.append("")
        
        # Trial statistics
        trials_df = self.get_trials_dataframe()
        if not trials_df.empty:
            report.append("Trial Statistics:")
            report.append(f"  Mean Score: {trials_df['value'].mean():.4f}")
            report.append(f"  Std Score: {trials_df['value'].std():.4f}")
            report.append(f"  Min Score: {trials_df['value'].min():.4f}")
            report.append(f"  Max Score: {trials_df['value'].max():.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Summary report saved to {save_path}")
        
        return report_text
    
    def save_results(self, base_path: str = "optimization_results"):
        """Save all results and plots"""
        if self.study is None:
            self.logger.error("No study set for saving results")
            return
        
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save trials data
        trials_df = self.get_trials_dataframe()
        if not trials_df.empty:
            trials_df.to_csv(f"{base_path}/trials.csv", index=False)
        
        # Save summary report
        self.generate_summary_report(f"{base_path}/summary.txt")
        
        # Save plots
        try:
            self.plot_optimization_history(f"{base_path}/optimization_history.png")
            self.plot_parameter_importance(f"{base_path}/parameter_importance.png")
            self.plot_parallel_coordinate(f"{base_path}/parallel_coordinate.png")
            self.plot_contour(save_path=f"{base_path}/contour.png")
        except Exception as e:
            self.logger.warning(f"Could not save some plots: {e}")
        
        self.logger.info(f"All results saved to {base_path}")