"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return project root path"""
    return project_root


@pytest.fixture(scope="session")
def sample_data_path(project_root_path):
    """Return path to sample data"""
    sample_path = project_root_path / "data" / "sample" / "sample_data.csv"
    if not sample_path.exists():
        # Try alternative location
        sample_path = project_root_path / "et_intel" / "data" / "sample" / "sample_data.csv"
    return sample_path if sample_path.exists() else None

