from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

class SourceType(str, Enum):
    VAT = "vat"
    DEFINITION = "definition"
    RYCZALT = "ryczalt"
    RYCZALT_DEFINITION = "ryczalt_definition"

@dataclass
class RagDocument:
    page_content: str
    metadata: Dict[str, Any]
