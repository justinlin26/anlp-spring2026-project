#!/bin/bash
# Runs on the EC2 instance via user-data at first boot (as root).
# Placeholders __S3_BUCKET__ and __REPO_URL__ are substituted by aws/launch_aws.sh.
#
# Guarantees three safety protections:
#   (1) On any exit of this script, sync results/checkpoints to S3 and terminate self.
#   (2) Independent 48h kill switch (survives even if this script hangs).
#   (3) Budget alerts are set up once by the user; see aws/README.md.

set -eux
exec > >(tee -a /var/log/grpo-setup.log) 2>&1

S3_BUCKET="__S3_BUCKET__"
REPO_URL="__REPO_URL__"

# --- Instance metadata via IMDSv2 ---
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
get_meta() {
    curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        "http://169.254.169.254/latest/meta-data/$1"
}
INSTANCE_ID=$(get_meta instance-id)
REGION=$(get_meta placement/region)

# --- Safety net: unconditional terminate at +48h ---
# Runs in a detached background process so it outlives this script.
nohup bash -c "sleep 172800 && aws ec2 terminate-instances \
    --instance-ids $INSTANCE_ID --region $REGION" \
    &> /var/log/grpo-killswitch.log &
disown

# --- On-exit hook: sync to S3, then terminate ---
cleanup() {
    set +e
    cd /home/ubuntu/anlp-project 2>/dev/null
    aws s3 sync results/     "s3://$S3_BUCKET/grpo-results/"     --region "$REGION"
    aws s3 sync checkpoints/ "s3://$S3_BUCKET/grpo-checkpoints/" --region "$REGION"
    aws s3 cp /var/log/grpo-setup.log "s3://$S3_BUCKET/logs/grpo-setup-$INSTANCE_ID.log" --region "$REGION"
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
}
trap cleanup EXIT

# --- Clone + install + train, as the ubuntu user ---
# The DLAMI has a preinstalled conda 'pytorch' env with CUDA-matched torch.
sudo -u ubuntu -i bash <<UBUNTU_BLOCK
set -eux
cd ~
if [ ! -d anlp-project ]; then
    git clone "$REPO_URL" anlp-project
fi
cd anlp-project

source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch
pip install -r requirements.txt

bash run_grpo.sh
UBUNTU_BLOCK

# trap fires here → sync + terminate
