"""OpenSafety AI threat detection system."""

from detection.analyzer import ThreatAnalyzer, ThreatAssessment, ThreatSignal
from detection.fingerprint import (
    AdvancedFingerprinter,
    ContentFingerprint,
    ContentType,
    FingerprintMatch,
)
from detection.patterns import AttackPattern, SplitAttackDetector
from detection.tracker import RequestTracker, TrackedRequest

# ML-based detector
from detection.ml_detector import (
    ThreatDetector,
    ThreatScore,
    FeatureHasher,
    BloomFilter,
    CountMinSketch,
    get_detector,
    score_text,
)

# Attack gene pool
from detection.attack_genes import (
    AttackGene,
    GeneCategory,
    GenePool,
    get_gene_pool,
    get_all_patterns,
    ATTACK_GENES,
)

# Threat ontology and discovery
from detection.threat_ontology import (
    ThreatDomain,
    ThreatCategory,
    ThreatTechnique,
    ThreatVector,
    ThreatKnowledgeBase,
    AutonomousThreatGenerator,
    ThreatDiscoveryAgent,
    GeneratedThreat,
    DiscoveryReport,
    create_threat_ontology,
    run_threat_discovery,
    get_threat_taxonomy,
)

__all__ = [
    # Core detection
    "RequestTracker",
    "TrackedRequest",
    "ThreatAnalyzer",
    "ThreatSignal",
    "ThreatAssessment",
    "SplitAttackDetector",
    "AttackPattern",
    "AdvancedFingerprinter",
    "ContentFingerprint",
    "FingerprintMatch",
    "ContentType",
    # ML detector
    "ThreatDetector",
    "ThreatScore",
    "FeatureHasher",
    "BloomFilter",
    "CountMinSketch",
    "get_detector",
    "score_text",
    # Attack genes
    "AttackGene",
    "GeneCategory",
    "GenePool",
    "get_gene_pool",
    "get_all_patterns",
    "ATTACK_GENES",
    # Threat ontology
    "ThreatDomain",
    "ThreatCategory",
    "ThreatTechnique",
    "ThreatVector",
    "ThreatKnowledgeBase",
    "AutonomousThreatGenerator",
    "ThreatDiscoveryAgent",
    "GeneratedThreat",
    "DiscoveryReport",
    "create_threat_ontology",
    "run_threat_discovery",
    "get_threat_taxonomy",
]
