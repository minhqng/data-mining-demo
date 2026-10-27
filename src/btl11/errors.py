"""Typed, user-facing error hierarchy for the BTL11 pipeline."""


class BTL11Error(Exception):
    """Base class for expected, actionable failures."""


class ConfigError(BTL11Error):
    """The YAML configuration is missing, malformed or violates a locked contract."""


class DataError(BTL11Error):
    """The raw or processed dataset is missing, inconsistent or unsupported."""


class SchemaError(DataError):
    """The raw dataset does not satisfy the mandatory schema."""


class ConflictingFeaturesError(DataError):
    """The same track_id carries contradicting audio vectors."""


class LicenseGateError(BTL11Error):
    """The provenance/license gate blocks empirical work."""


class ArtifactError(BTL11Error):
    """A run artifact is missing, corrupted or has an unexpected schema."""


class RunError(BTL11Error):
    """A run cannot be created or completed safely."""


class VerificationError(BTL11Error):
    """An artifact failed independent verification."""
