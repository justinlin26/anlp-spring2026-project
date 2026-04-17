# AWS launch for GRPO training

Runs `run_grpo.sh` on a single-GPU EC2 instance, syncs results to S3, and auto-terminates.

## What's built in

1. **Auto-terminate on training completion** — `setup_instance.sh` traps EXIT, syncs `results/` and `checkpoints/` to S3, then calls `ec2:TerminateInstances` on itself.
2. **48h hard kill switch** — a detached background process runs `sleep 172800 && terminate-instances` at boot, independent of the training script. Triggers even if training hangs.
3. **Budget alert** — one-time CLI setup below.

S3 is used (not just EBS) because the EBS volume is destroyed on terminate. Checkpoints/results must be uploaded before that happens.

## One-time setup

You'll need AWS CLI configured (`aws configure`) and the following created once. Copy-paste the commands below, filling in `MY_BUCKET` etc.

```bash
export REGION=us-east-1
export MY_BUCKET=anlp-grpo-$USER        # pick a globally-unique name
export KEY_NAME=anlp-grpo-key
```

### 1. S3 bucket

```bash
aws s3api create-bucket --bucket $MY_BUCKET --region $REGION \
    $([ "$REGION" != us-east-1 ] && echo "--create-bucket-configuration LocationConstraint=$REGION")
```

### 2. SSH key pair

```bash
aws ec2 create-key-pair --key-name $KEY_NAME --region $REGION \
    --query "KeyMaterial" --output text > ~/.ssh/$KEY_NAME.pem
chmod 400 ~/.ssh/$KEY_NAME.pem
```

### 3. Security group (SSH-only)

```bash
SG_ID=$(aws ec2 create-security-group --group-name anlp-grpo-sg \
    --description "GRPO training SSH access" --region $REGION \
    --query "GroupId" --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
    --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32 \
    --region $REGION
echo "SECURITY_GROUP=$SG_ID"
```

### 4. IAM instance profile

The instance needs permission to (a) terminate itself and (b) write to S3.

```bash
# Trust policy: EC2 can assume this role
cat > /tmp/trust.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Permission policy: terminate only instances tagged Project=anlp-11711, write to our bucket
cat > /tmp/perms.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ec2:TerminateInstances",
      "Resource": "*",
      "Condition": {"StringEquals": {"ec2:ResourceTag/Project": "anlp-11711"}}
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::$MY_BUCKET", "arn:aws:s3:::$MY_BUCKET/*"]
    }
  ]
}
EOF

aws iam create-role --role-name grpo-training-role \
    --assume-role-policy-document file:///tmp/trust.json
aws iam put-role-policy --role-name grpo-training-role \
    --policy-name grpo-training-policy \
    --policy-document file:///tmp/perms.json
aws iam create-instance-profile --instance-profile-name grpo-training-profile
aws iam add-role-to-instance-profile \
    --instance-profile-name grpo-training-profile \
    --role-name grpo-training-role
```

### 5. Budget alert (so you know when credits are burning)

Set a $100 monthly budget with email alerts at 50/80/95%. Replace `YOUR_EMAIL` and `ACCOUNT_ID`.

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export YOUR_EMAIL=you@andrew.cmu.edu

cat > /tmp/budget.json <<EOF
{
  "BudgetName": "anlp-grpo-monthly",
  "BudgetLimit": {"Amount": "100", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF

cat > /tmp/notifs.json <<EOF
[
  {
    "Notification": {"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":50,"ThresholdType":"PERCENTAGE"},
    "Subscribers": [{"SubscriptionType":"EMAIL","Address":"$YOUR_EMAIL"}]
  },
  {
    "Notification": {"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80,"ThresholdType":"PERCENTAGE"},
    "Subscribers": [{"SubscriptionType":"EMAIL","Address":"$YOUR_EMAIL"}]
  },
  {
    "Notification": {"NotificationType":"FORECASTED","ComparisonOperator":"GREATER_THAN","Threshold":95,"ThresholdType":"PERCENTAGE"},
    "Subscribers": [{"SubscriptionType":"EMAIL","Address":"$YOUR_EMAIL"}]
  }
]
EOF

aws budgets create-budget --account-id $ACCOUNT_ID \
    --budget file:///tmp/budget.json \
    --notifications-with-subscribers file:///tmp/notifs.json
```

## Launch a training run

```bash
export S3_BUCKET=$MY_BUCKET
export KEY_NAME=$KEY_NAME
export SECURITY_GROUP=$SG_ID
export AWS_REGION=$REGION
bash aws/launch_aws.sh
```

Default instance is `g6e.2xlarge` (1×L40S 48GB, ~$2.50/hr). Override with `INSTANCE_TYPE=p4d.24xlarge` for 8×A100 if you have quota and want the exact target.

## Monitor / intervene

- **Tail the training log**: `ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@<public-ip> 'sudo tail -f /var/log/grpo-setup.log'`
- **Check S3 progress**: `aws s3 ls s3://$MY_BUCKET/grpo-results/ --recursive`
- **Kill the instance early**: `aws ec2 terminate-instances --instance-ids i-xxx --region $REGION`

## Cost estimate

`g6e.2xlarge` at $2.50/hr × 48h = ~$120 max per full run. Most runs finish well before 48h.
