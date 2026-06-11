# =============================================================
# SageMaker Deployment Script
# Deploys both regression and classification models as endpoints
# =============================================================
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import time

# ── Config — paste your role ARN here ────────────────────────
ROLE_ARN      = "arn:aws:iam::207635141795:role/SageMakerExecutionRole"
BUCKET        = "alanantony-realestate-ml"
REGION        = "us-east-1"

# S3 paths to your uploaded model artifacts
REGRESSION_S3     = f"s3://{BUCKET}/models/regression/rf_regressor.tar.gz"
CLASSIFICATION_S3 = f"s3://{BUCKET}/models/classification/logistic_classifier.tar.gz"

# ── Session ───────────────────────────────────────────────────
boto3.setup_default_session(region_name=REGION)
sagemaker_session = sagemaker.Session()

# ── Deploy Regression Model ───────────────────────────────────
#print("Deploying regression model...")
#regression_model = SKLearnModel(
#    model_data=REGRESSION_S3,
#    role=ROLE_ARN,
#    entry_point="inference.py",
#    framework_version="0.23-1",
#    py_version="py3",
#    sagemaker_session=sagemaker_session,
#)

#regression_predictor = regression_model.deploy(
#    initial_instance_count=1,
#    instance_type="ml.t2.medium",
#    endpoint_name="realestate-regression",
#)
#print(f"Regression endpoint deployed: realestate-regression")

# ── Deploy Classification Model ───────────────────────────────
print("\nDeploying classification model...")
classification_model = SKLearnModel(
    model_data=CLASSIFICATION_S3,
    role=ROLE_ARN,
    entry_point="inference.py",
    framework_version="0.23-1",
    py_version="py3",
    sagemaker_session=sagemaker_session,
)

classification_predictor = classification_model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name="realestate-classification",
)
print(f"Classification endpoint deployed: realestate-classification")

# ── Test both endpoints ───────────────────────────────────────
print("\nTesting regression endpoint...")
test_input = "8.3252,41.0,6.984,1.023,322.0,2.555,37.88,-122.23"
regression_predictor.serializer   = sagemaker.serializers.CSVSerializer()
regression_predictor.deserializer = sagemaker.deserializers.StringDeserializer()
result = regression_predictor.predict(test_input)
print(f"Regression prediction: ${float(result) * 100_000:,.0f}")

print("\nTesting classification endpoint...")
test_input = "35,management,married,university.degree,no,yes,no,cellular,may,mon,300,2,999,0,nonexistent,-1.8,92.893,-46.2,1.299,5099.1"
classification_predictor.serializer   = sagemaker.serializers.CSVSerializer()
classification_predictor.deserializer = sagemaker.deserializers.StringDeserializer()
result = classification_predictor.predict(test_input)
print(f"Classification prediction: {result}")

print("\nBoth endpoints are live and ready!")
print("Endpoint names:")
print("  realestate-regression")
print("  realestate-classification")
