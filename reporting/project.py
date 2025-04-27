import time
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.ui.workspace import Workspace,WorkspaceBase
import logging
import os


logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Include timestamp and log level
    handlers=[
        logging.StreamHandler()  # Print logs to the console
    ]
)

logging.info("LAUNCHING REPORT")

WORKSPACE = "workspace"

PROJECT_NAME = "ML Project"
PROJECT_DESCRIPTION = "Machine Learning Image Classification Project"


current_data = pd.read_csv("./data/prod_data.csv")
reference_data = pd.read_csv("./data/ref_data.csv")

column_mapping = ColumnMapping(
    target="target",
    prediction="prediction",
)


report = Report(metrics=[ClassificationPreset(),DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

def create_project(workspace : WorkspaceBase):
    project = workspace.create_project(PROJECT_NAME)
    project.description = PROJECT_DESCRIPTION
    project.save()
    return project

workspace = Workspace.create(WORKSPACE)
project = create_project(workspace)
workspace.add_report(project.id, report)

logging.info("REPORT AVAILABLE")
