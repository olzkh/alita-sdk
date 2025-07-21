from enum import Enum
from pydantic import BaseModel, Field


class CarrierCredentials(BaseModel):
    """Data model for the credentials required to configure the CarrierClient."""
    url: str
    token: str
    organization: str
    project_id: str


class ReportType(str, Enum):
    backend = 'backend'
    ui = 'ui'


class ReportRequest(BaseModel):
    report_id: int = Field(..., description="The unique report ID")
    report_type: ReportType = Field(..., description="Type of report: backend or ui")
