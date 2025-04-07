import os
import sys
import paramiko
from dotenv import load_dotenv
from colorama import init, Fore, Style

init(autoreset=True)

# === Color helpers ===
def red(text): return Fore.LIGHTRED_EX + text + Style.RESET_ALL
def green(text): return Fore.LIGHTGREEN_EX + text + Style.RESET_ALL
def yellow(text): return Fore.YELLOW + text + Style.RESET_ALL

# === Load env ===
load_dotenv()

def load_env_vars():
    try:
        env = {
            "SSH_HOST": os.environ["SSH_HOST"],
            "SSH_PORT": int(os.environ.get("SSH_PORT", 22)),
            "SSH_USER": os.environ["SSH_USER"],
            "PASSWORD": os.environ["PASSWORD"],  # for sudo
            "SSH_KEY_PATH": os.environ["SSH_KEY_PATH"],
            "ARGOCD_APP": os.environ["ARGOCD_APP"],
            "ARGOCD_USER": os.environ["ARGOCD_USER"],
            "ARGOCD_USER_PASS": os.environ["ARGOCD_USER_PASS"],
            "DEPLOYMENT_NAME": os.environ["DEPLOYMENT_NAME"],
            "NAMESPACE": os.environ["NAMESPACE"],
            "LOCAL_FILE": os.environ["LOCAL_FILE"],
            "HOST_VOLUME_PATH": os.environ["HOST_VOLUME_PATH"],
            "ARGOCD_PORT": os.getenv("ARGOCD_PORT", "80"),
        }

        for file_var in ["LOCAL_FILE", "SSH_KEY_PATH"]:
            if not os.path.isfile(env[file_var]):
                print(red(f"[✗] File not found: {env[file_var]}"))
                sys.exit(1)

        return env
    except KeyError as e:
        print(red(f"[✗] Missing required environment variable: {e}"))
        sys.exit(1)

env = load_env_vars()

# === SSH helpers ===
def ssh_connect():
    try:
        key = paramiko.RSAKey.from_private_key_file(env["SSH_KEY_PATH"])
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=env["SSH_HOST"], port=env["SSH_PORT"], username=env["SSH_USER"], pkey=key)
        return ssh
    except Exception as e:
        print(red(f"[✗] SSH connection failed: {e}"))
        sys.exit(1)

def ssh_exec(ssh, command):
    print(yellow(f"[*] Running: {command}"))
    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    if error:
        print(red(f"[stderr] {error}"))
    return output, error

def ssh_exec_sudo(ssh, command, password):
    full_command = f"sudo -S {command}"
    stdin, stdout, stderr = ssh.exec_command(full_command, get_pty=True)
    stdin.write(password + '\n')
    stdin.flush()
    out = stdout.read().decode()
    err = stderr.read().decode()

    # Strip sudo prompt if it shows up
    if out.startswith("[sudo] password"):
        out = '\n'.join(out.split('\n')[1:])

    if err:
        print(red(f"[stderr] {err}"))
    return out.strip(), err.strip()

def ssh_copy_file(ssh, local_file, remote_path):
    try:
        sftp = ssh.open_sftp()
        print(yellow(f"[*] Copying '{local_file}' to '{remote_path}'..."))
        sftp.put(local_file, remote_path)
        sftp.close()
        print(green("[✓] File copied successfully."))
    except Exception as e:
        print(red(f"[✗] File copy failed: {e}"))
        sys.exit(1)

# === Main ===
if __name__ == "__main__":
    yaml_path = env["POD_YAML_FILE"]
    if not os.path.isfile(yaml_path):
        print(red(f"[✗] Pod YAML file not found: {yaml_path}"))
        sys.exit(1)

    ssh = ssh_connect()
    print(green(f"=== Connected to {env['SSH_HOST']} ==="))

    # Step 0: Get ArgoCD service ClusterIP
    remote_ip, err = ssh_exec_sudo(
        ssh,
        "kubectl get svc argocd-server -n argocd -o jsonpath='{.spec.clusterIP}'",
        env["PASSWORD"]
    )
    if not remote_ip:
        print(red("[✗] Failed to get ArgoCD service IP"))
        sys.exit(1)

    # Step 1: Login to ArgoCD (in-cluster IP)
    ssh_exec(
        ssh,
        f"argocd login {remote_ip}:{env['ARGOCD_PORT']} --username {env['ARGOCD_USER']} --password {env['ARGOCD_USER_PASS']} --insecure"
    )

    # Step 2: Disable sync
    ssh_exec(ssh, f"argocd app set {env['ARGOCD_APP']} --sync-policy none")
    print(green(f"[✓] Sync disabled for app '{env['ARGOCD_APP']}'"))

    # Step 3: Scale down deployment
    ssh_exec(ssh, f"kubectl -n {env['NAMESPACE']} scale deployment {env['DEPLOYMENT_NAME']} --replicas=0")
    print(green(f"[✓] Deployment '{env['DEPLOYMENT_NAME']}' scaled down"))

    # Step 4: Apply pod YAML from file
    print(yellow(f"[*] Applying pod manifest from {yaml_path}..."))
    ssh_exec(ssh, f"kubectl apply -f {yaml_path} -n {env['NAMESPACE']}")
    pod_name = "pvc-copy-helper"  # should match YAML
    print(yellow(f"[*] Waiting for pod '{pod_name}' to be ready..."))
    ssh_exec(ssh, f"kubectl -n {env['NAMESPACE']} wait pod/{pod_name} --for=condition=Ready --timeout=60s")
    print(green("Temporary pod is ready and running"))

    # Step 5: Copy file to PVC
    remote_target = f"{env['NAMESPACE']}/{pod_name}:{env['HOST_VOLUME_PATH']}"
    scp_cmd = f"kubectl cp {env['LOCAL_FILE']} {remote_target} -n {env['NAMESPACE']}"
    ssh_exec(ssh, scp_cmd)
    print(green(f"[✓] File copied into PVC via pod '{pod_name}'"))

    # Step 6: Remove temp pod
    ssh_exec(ssh, f"kubectl -n {env['NAMESPACE']} delete pod {pod_name}")
    print(green(f"[✓] Temporary pod '{pod_name}' deleted"))
    
    # Step 7: Scale up deployment
    ssh_exec(ssh, f"kubectl -n {env['NAMESPACE']} scale deployment {env['DEPLOYMENT_NAME']} --replicas=1")
    print(green(f"[✓] Deployment '{env['DEPLOYMENT_NAME']}' scaled up"))

    # Step 8: Re-enable ArgoCD sync
    ssh_exec(ssh, f"argocd app set {env['ARGOCD_APP']} --sync-policy automated")
    print(green(f"[✓] Sync re-enabled for app '{env['ARGOCD_APP']}'"))

    ssh.close()
    print(green("=== SSH connection closed ==="))

