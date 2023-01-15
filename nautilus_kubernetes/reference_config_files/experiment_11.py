import os
import subprocess
import time
import sqlite3

EXPERIMENT_NAME = "experiment_11"

# kubernets job schema:
schema = """apiVersion: batch/v1
kind: Job
metadata:
  name: {0}
  namespace: rd
spec: 
  template:
    spec:
      nodeSelector:
        nautilus.io/disktype: nvme 
      containers:
      - name: get-features
        image: somil55/condapytorch:v1.0.2
        command: ["conda", "run", "-n", "netdissectlite", "/bin/bash", "-c", "/root/ddivyanshcephfspvc/ExplainableAI/nautilus_start_script.sh && cd /root/ddivyanshcephfspvc/ExplainableAI && {1} && /root/ddivyanshcephfspvc/ExplainableAI/nautilus_stop_script.sh"]
        imagePullPolicy: Always
        resources:
          limits:
            memory: "48Gi"
            cpu: 6
            nvidia.com/gpu: 1
            ephemeral-storage: 500Gi
          requests:
            memory: "32Gi"
            cpu: 6
            nvidia.com/gpu: 1
            ephemeral-storage: 500Gi
        volumeMounts:
        - mountPath: /root/ddivyanshcephfspvc
          name: ddivyanshcephfspvc
        - mountPath: /root/ddivyanshcephfspvcoutput/
          name: ddivyanshcephfspvcoutputexp
        - name: data
          mountPath: /mnt/data
      volumes:
        - name: ddivyanshcephfspvc
          persistentVolumeClaim:
            claimName: ddivyanshcephfspvc
        - name: ddivyanshcephfspvcoutputexp
          persistentVolumeClaim:
            claimName: ddivyanshcephfspvcoutputexp
        - name: data
          emptyDir: {{}}
      restartPolicy: Never
  backoffLimit: 2
"""

def is_job_running(job_name):
    # check if kubectl job with name job_name is starting
    output = subprocess.check_output(["kubectl", "get", "pods"]).decode("utf-8")

    # check at least one pod that starts with job_name is starting
    for line in output.split("\n"):
        if job_name in line:
            print("Job {} is in state {}".format(job_name, line))

    return False

def main():
    experiment_dir = os.path.join("experiments", EXPERIMENT_NAME)

    # create experiment directory, if it doesn't exist
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    for i in range(50):
      # create file name for the experiment
      file_name = f"{EXPERIMENT_NAME}_{i}.yaml"

      # setup job name
      job_name = "{}-{}".format(EXPERIMENT_NAME, i)
      job_name = job_name.replace(".", "-").replace("_", "-")

      # get commands
      command = "python random_target_resnet50_robust.py"

      # save file
      with open(os.path.join(experiment_dir, file_name), "w") as f:
          f.write(schema.format(job_name, command))

      if is_job_running(job_name):
        continue

      # delete existing job
      os.system("kubectl delete job {}".format(job_name))

      # # run job
      # os.system("kubectl create -f {}".format(os.path.join(experiment_dir, file_name)))

      # # sleep for 5 seconds
      # time.sleep(5)  

            
if __name__ == "__main__":
    main()
