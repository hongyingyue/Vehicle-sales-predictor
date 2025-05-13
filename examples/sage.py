import sagemaker
from sagemaker.model import Model
from sagemaker import get_execution_role

role = get_execution_role()
sess = sagemaker.Session()

model = Model(
    image_uri="<your_account_id>.dkr.ecr.us-west-2.amazonaws.com/vehicle-sales-api:latest",
    role=role,
    sagemaker_session=sess,
)

predictor = model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name="vehicle-sales-endpoint")
