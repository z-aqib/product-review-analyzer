# product-review-analyzer
this is a product review analyzer multi modal project for course mlops

Milestone 1 verifies that each project team has translated its idea into a production-ready
repository skeleton.

## STEPS TO RUN THE PROJECT LOCALLY

### MILESTONE 1
first create a virtual environment
```bash
python -m venv mlops-venv
.mlops-venv\Scripts\activate
```
then install the requirements
```bash
pip install -r requirements.txt
```
then activate the pre-commit hooks
```bash
pre-commit install
```

D9 Cloud Setup
aws configure

AWS Access Key ID [None]: provided
AWS Secret Access Key [None]: provided
Default region name [None]: us-east-1
Default output format [None]: json

Bucket creation command (can change name)
aws s3 mb s3://mlops-d9 --region us-east-1

Bucket list command
aws s3 ls

Command to upload to S3
aws s3 cp README.md s3://mlops-d9/ 

Command to check what has been uploaded to S3 Bucket
aws s3 ls s3://mlops-d9/

EC2
cd into the directory where the mlops-key pem file is located

Command to connect to EC2 instance (this opens ubuntu)
ssh -i "mlops-key.pem" ubuntu@ec2-13-60-193-55.eu-north-1.compute.amazonaws.com

Public IPv4 address 13.60.193.55
Private IPv4 addresses 172.31.38.0
Public DNS ec2-13-60-193-55.eu-north-1.compute.amazonaws.com

Do this in ubuntu terminal

Clone your GitHub repo
Install dependencies(do pip install boto3,
ADD THESE 3 LINES IN APP.PY
>import boto3

>s3 = boto3.client('s3')
>s3.download_file('your-bucket-name', 'model.pkl', 'model.pkl'))
Run your model’s FastAPI/Flask/Streamlit serve

if mlops-key.pem gets a warning for unsafe, run this with correct path, in powershell as administrator
icacls "D:\sem7\mlops\project\product-review-analyzer\mlops-key.pem" /inheritance:r
icacls "D:\sem7\mlops\project\product-review-analyzer\mlops-key.pem" /grant:r "$($env:USERNAME):(R)"

then back in vscode run this, run with correct path
ssh -i "D:\sem7\mlops\project\product-review-analyzer\mlops-key.pem" ubuntu@ec2-13-60-193-55.eu-north-1.compute.amazonaws.com

Uploaded processed data using this command
aws s3 cp . s3://mlops-d9/processed/ --recursive

Checked if data is uploaded using this
aws s3 ls s3://mlops-d9/processed/

Command to delete something from S3
aws s3 rm s3://mlops-d9/processed/README.md

If fastapi app not running check if port is busy
netstat -ano | findstr :8000

Command to kill any task
taskkill /PID 20112 /F

Run this to avoid pre-commit
git commit -m "Your commit message" --no-verify

Run when something changed in main.py
docker build -t product-recommender .

Run a new container
docker run -d -p 8000:8000 --name recommender product-recommender

Shows running
docker logs recommender

Force stop container
docker stop recommender
docker rm recommender

Health checkpoint
http://127.0.0.1:8000/health

Use to check if backend working
curl http://127.0.0.1:8000/recommend?user_id=AE243IWFZJ3BB6E6WMUG52DHWJVA&k=5

When you want to run with prometheus and grafana
docker-compose up --build

Prometheus runs on this
http://localhost:9090

Grafana runs on this
http://localhost:3000

Grafana username: admin, pw: admin