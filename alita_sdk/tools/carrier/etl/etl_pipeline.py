"""
ETL Pipeline Interfaces

This module defines the core interfaces for the Extract, Transform, Load (ETL) pipeline,
ensuring proper separation of concerns and adherence to SOLID principles.

Author: Karen Florykian
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, context: Dict[str, Any]) -> Any:
        """Extracts raw data from a source (e.g., Carrier API). Returns data structure."""
        pass


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, extracted_data: Any, context: Dict[str, Any]) -> Any:
        """Transforms raw data into the desired final format. Returns transformed data."""
        pass


class BaseLoader(ABC):
    @abstractmethod
    def load(self, transformed_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Loads the final data to a destination (e.g., Carrier bucket). Returns result dictionary."""
        pass


class ETLPipeline:
    """A generic pipeline that executes the E-T-L steps in sequence."""

    def __init__(self, extractor: BaseExtractor, transformer: BaseTransformer, loader: BaseLoader):
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        self.summary = {}

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the full pipeline."""
        extracted_data = self.extractor.extract(context)
        context.update(extracted_data)
        transformed_data = self.transformer.transform(extracted_data, context)
        load_result = self.loader.load(transformed_data, context)
        self.summary = load_result
        return self.summary

    def get_run_summary(self, error: Exception = None) -> dict:
        if error:
            self.summary['status'] = 'FAILED'
            self.summary['error'] = str(error)
        return self.summary
