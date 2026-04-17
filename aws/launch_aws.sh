#!/bin/bash
# Launch a GPU EC2 instance that clones, trains GRPO, syncs results to S3, and auto-terminates.
# Prereqs (one-time, see aws/README.md): IAM instance profile, SSH key pair, security group, S3 bucket.
#
# Usage:
#   S3_BUCKET=my-anlp-bucket KEY_NAME=my-key SECURITY_GROUP=sg-xxx bash aws/launch_aws.sh

set -eu

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.2xlarge}"    # 1x L40S 48GB, ~$2.5/hr. Alt: p4d.24xlarge (8xA100 40GB, ~$32/hr).
S3_BUCKET="${S3_BUCKET:-}"
KEY_NAME="${KEY_NAME:-}"
SECURITY_GROUP="${SECURITY_GROUP:-}"
IAM_PROFILE="${IAM_PROFILE:-grpo-training-profile}"
REPO_URL="${REPO_URL:-https://github.com/justinlin26/anlp-spring2026-project.git}"
DISK_GB="${DISK_GB:-200}"

for var in S3_BUCKET KEY_NAME SECURITY_GROUP; do
    if [ -z "${!var}" ]; then
        echo "ERROR: env var $var is required. See aws/README.md for one-time setup." >&2
        exit 1
    fi
done

# Latest Deep Learning AMI (Ubuntu 22.04, CUDA + PyTorch preinstalled)
AMI_ID=$(aws ssm get-parameter \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
    --region "$REGION" \
    --query "Parameter.Value" --output text)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DATA=$(mktemp)
sed -e "s|__S3_BUCKET__|$S3_BUCKET|g" \
    -e "s|__REPO_URL__|$REPO_URL|g" \
    "$SCRIPT_DIR/setup_instance.sh" > "$USER_DATA"

echo "Launching $INSTANCE_TYPE in $REGION (AMI $AMI_ID)..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\"}}]" \
    --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
    --user-data "file://$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=grpo-training},{Key=Project,Value=anlp-11711}]" \
    --region "$REGION" \
    --query "Instances[0].InstanceId" --output text)
rm -f "$USER_DATA"

echo "Launched $INSTANCE_ID"
echo "Auto-terminates on training completion OR at +48h (whichever is first)."
echo ""
echo "Tail setup log:  ssh ubuntu@\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].PublicIpAddress' --output text) 'sudo tail -f /var/log/grpo-setup.log'"
echo "Watch results:   aws s3 ls s3://$S3_BUCKET/grpo-results/ --recursive"
echo "Kill early:      aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
