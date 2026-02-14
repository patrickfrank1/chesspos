#!/bin/bash

cd /home/studio-lab-user/sagemaker-studiolab-notebooks/chess-embedding-experiments/sagemaker

# Load environment variables from .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found."
    exit 1
fi

# set up git
git config --global user.email "$GITHUB_EMAIL"
git config --global user.name "$GITHUB_SIGNATURE_NAME"
echo "https://$GITHUB_USER_NAME:$GITHUB_TOKEN@github.com/$GITHUB_REPO_OWNER/$GITHUB_REPO_NAME.git"
git remote remove origin
git remote add origin "https://$GITHUB_USER_NAME:$GITHUB_TOKEN@github.com/$GITHUB_REPO_OWNER/$GITHUB_REPO_NAME.git"

# set up dvc
dvc remote add colab_origin s3://dvc
dvc remote modify colab_origin endpointurl "https://dagshub.com/$DAGSHUB_REPO_OWNER/$DAGSHUB_REPO_NAME.s3"
dvc remote modify colab_origin --local access_key_id "$DAGSHUB_TOKEN"
dvc remote modify colab_origin --local secret_access_key "$DAGSHUB_TOKEN"
dvc pull -r colab_origin