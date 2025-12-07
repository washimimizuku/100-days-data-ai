"""
Data Quality Framework - Main Implementation
Integrates profiling, validation, testing, scoring, and reporting
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime
from scipy import stats


class DataProfiler:
    """Profile datasets to understand characteristics"""
    
    def profile(self, df):
        """Generate comprehensive profile"""
        profile = {
            'overview': self._profile_overview(df),
            'numerical': self._profile_numerical_columns(df),
            'categorical': self._profile_categorical_columns(df),
            'missing': self._analyze_missing(df),
            'correlations': self._analyze_correlations(df),
            'issues': []
        }
        
        # Identify issues
        profile['issues'] = self._identify_issues(profile)
        
        return profile
    
    def _profile_overview(self, df):
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': df.duplicated().sum()
        }
    
    def _profile_numerical_columns(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        profiles = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            
            profiles[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'outliers': len(outliers),
                'zeros': (df[col] == 0).sum()
            }
        
        return profiles
    
    def _profile_categorical_columns(self, df):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        profiles = {}
        
        for col in categorical_cols:
            profiles[col] = {
                'unique': df[col].nunique(),
                'cardinality': df[col].nunique() / len(df),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        return profiles
    
    def _analyze_missing(self, df):
        missing = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing[col] = {
                    'count': missing_count,
                    'percentage': missing_count / len(df) * 100
                }
        return missing
    
    def _analyze_correlations(self, df):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return []
        
        corr_matrix = df[numerical_cols].corr()
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_corr.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return high_corr
    
    def _identify_issues(self, profile):
        issues = []
        
        # Check for high missing rates
        for col, stats in profile['missing'].items():
            if stats['percentage'] > 10:
                issues.append(f"{col}: {stats['percentage']:.1f}% missing")
        
        # Check for outliers
        for col, stats in profile['numerical'].items():
            if stats['outliers'] > 0:
                issues.append(f"{col}: {stats['outliers']} outliers detected")
        
        # Check for duplicates
        if profile['overview']['duplicates'] > 0:
            issues.append(f"{profile['overview']['duplicates']} duplicate rows")
        
        return issues


class DataValidator:
    """Validate data against rules"""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule):
        """Add validation rule"""
        self.rules.append(rule)
    
    def validate(self, df):
        """Run all validation rules"""
        results = []
        
        for rule in self.rules:
            result = rule.validate(df)
            results.append(result)
        
        passed = all(r['passed'] for r in results)
        
        return {
            'passed': passed,
            'total_rules': len(results),
            'passed_rules': sum(1 for r in results if r['passed']),
            'failed_rules': sum(1 for r in results if not r['passed']),
            'results': results
        }


class ValidationRule:
    """Base validation rule"""
    
    def __init__(self, name, column=None):
        self.name = name
        self.column = column
    
    def validate(self, df):
        raise NotImplementedError


class NotNullRule(ValidationRule):
    """Validate no nulls"""
    
    def __init__(self, column):
        super().__init__(f"{column}_not_null", column)
    
    def validate(self, df):
        null_count = df[self.column].isna().sum()
        return {
            'rule': self.name,
            'passed': null_count == 0,
            'message': f"Found {null_count} nulls" if null_count > 0 else "No nulls",
            'invalid_count': null_count
        }


class UniqueRule(ValidationRule):
    """Validate uniqueness"""
    
    def __init__(self, column):
        super().__init__(f"{column}_unique", column)
    
    def validate(self, df):
        dup_count = df[self.column].duplicated().sum()
        return {
            'rule': self.name,
            'passed': dup_count == 0,
            'message': f"Found {dup_count} duplicates" if dup_count > 0 else "All unique",
            'invalid_count': dup_count
        }


class RangeRule(ValidationRule):
    """Validate value range"""
    
    def __init__(self, column, min_val, max_val):
        super().__init__(f"{column}_range", column)
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, df):
        invalid = df[(df[self.column] < self.min_val) | (df[self.column] > self.max_val)]
        return {
            'rule': self.name,
            'passed': len(invalid) == 0,
            'message': f"Found {len(invalid)} outside [{self.min_val}, {self.max_val}]",
            'invalid_count': len(invalid)
        }


class QualityScorer:
    """Calculate quality scores"""
    
    def __init__(self, weights=None):
        self.weights = weights or {
            'accuracy': 0.20,
            'completeness': 0.20,
            'consistency': 0.15,
            'validity': 0.20,
            'uniqueness': 0.15,
            'timeliness': 0.10
        }
    
    def calculate_score(self, df, validation_results):
        """Calculate overall quality score"""
        dimensions = {
            'completeness': self._score_completeness(df),
            'validity': self._score_validity(validation_results),
            'uniqueness': self._score_uniqueness(df)
        }
        
        # Calculate weighted score
        overall = sum(dimensions[dim] * self.weights.get(dim, 0) for dim in dimensions)
        
        # Normalize to available dimensions
        total_weight = sum(self.weights.get(dim, 0) for dim in dimensions)
        if total_weight > 0:
            overall = overall / total_weight
        
        grade = self._assign_grade(overall)
        
        return {
            'overall': overall,
            'grade': grade,
            'dimensions': dimensions
        }
    
    def _score_completeness(self, df):
        """Score based on missing data"""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        return 1 - (missing_cells / total_cells)
    
    def _score_validity(self, validation_results):
        """Score based on validation results"""
        if validation_results['total_rules'] == 0:
            return 1.0
        return validation_results['passed_rules'] / validation_results['total_rules']
    
    def _score_uniqueness(self, df):
        """Score based on duplicates"""
        dup_count = df.duplicated().sum()
        return 1 - (dup_count / len(df))
    
    def _assign_grade(self, score):
        """Convert score to letter grade"""
        if score >= 0.95: return 'A'
        elif score >= 0.85: return 'B'
        elif score >= 0.75: return 'C'
        elif score >= 0.65: return 'D'
        else: return 'F'


class QualityReporter:
    """Generate quality reports"""
    
    def generate_report(self, df, profile, validation, score):
        """Generate comprehensive report"""
        report = f"""
{'='*70}
DATA QUALITY REPORT
{'='*70}

Dataset Overview:
  Rows:        {profile['overview']['rows']:,}
  Columns:     {profile['overview']['columns']}
  Memory:      {profile['overview']['memory_mb']:.2f} MB
  Duplicates:  {profile['overview']['duplicates']}

{'='*70}
Quality Score: {score['overall']:.1%} (Grade: {score['grade']})
{'='*70}

Dimension Scores:
"""
        
        for dim, value in score['dimensions'].items():
            report += f"  {dim.capitalize():15} {value:.1%}\n"
        
        report += f"""
{'='*70}
Validation Results:
{'='*70}
  Total Rules:   {validation['total_rules']}
  Passed:        {validation['passed_rules']}
  Failed:        {validation['failed_rules']}

"""
        
        if validation['failed_rules'] > 0:
            report += "Failed Rules:\n"
            for result in validation['results']:
                if not result['passed']:
                    report += f"  ❌ {result['rule']}: {result['message']}\n"
        
        report += f"""
{'='*70}
Quality Issues ({len(profile['issues'])}):
{'='*70}
"""
        
        for issue in profile['issues'][:10]:
            report += f"  • {issue}\n"
        
        report += f"\n{'='*70}\n"
        
        return report


class QualityFramework:
    """Main quality framework orchestrator"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.profiler = DataProfiler()
        self.validator = DataValidator()
        self.scorer = QualityScorer()
        self.reporter = QualityReporter()
    
    def process(self, df):
        """Run complete quality pipeline"""
        # Profile data
        profile = self.profiler.profile(df)
        
        # Validate data
        validation = self.validator.validate(df)
        
        # Calculate score
        score = self.scorer.calculate_score(df, validation)
        
        # Generate report
        report = self.reporter.generate_report(df, profile, validation, score)
        
        # Apply quality gates
        valid_df, invalid_df = self._apply_gates(df, validation, score)
        
        return valid_df, invalid_df, {
            'profile': profile,
            'validation': validation,
            'score': score,
            'report': report
        }
    
    def _apply_gates(self, df, validation, score):
        """Apply quality gates and quarantine invalid data"""
        gates_config = self.config.get('gates', {})
        
        if not gates_config.get('enabled', True):
            return df, pd.DataFrame()
        
        min_score = gates_config.get('min_score', 0.70)
        
        if score['overall'] < min_score:
            print(f"⚠️  Quality score {score['overall']:.1%} below threshold {min_score:.1%}")
        
        # Collect invalid row indices
        invalid_indices = set()
        
        for result in validation['results']:
            if not result['passed'] and result.get('invalid_count', 0) > 0:
                # This is simplified; in practice, track actual indices
                pass
        
        # Split data
        if gates_config.get('quarantine_invalid', True) and len(invalid_indices) > 0:
            valid_df = df[~df.index.isin(invalid_indices)]
            invalid_df = df[df.index.isin(invalid_indices)]
        else:
            valid_df = df
            invalid_df = pd.DataFrame()
        
        return valid_df, invalid_df



