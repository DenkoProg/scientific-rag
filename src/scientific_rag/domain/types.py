from enum import StrEnum


class DataSource(StrEnum):
    ARXIV = "arxiv"
    PUBMED = "pubmed"


class SectionType(StrEnum):
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    CONCLUSION = "conclusion"
    ABSTRACT = "abstract"
    OTHER = "other"
