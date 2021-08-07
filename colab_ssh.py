#! /usr/bin/env python3
import json
import os
import sys
import subprocess


def get_json(file_path):
    file_path = os.path.abspath(file_path)
    with open(file_path) as f:
        data = json.load(f)

    return data


def get_ssh_host(host_name):
    return f'https://{host_name}.trycloudflare.com'


def get_exe(data, host_name):
    ports = []
    for key, value in data.items():
        if key.endswith('_port'):
            ports.append(value)
    ip = data['ip']

    forwards = ' '.join([f'-L {port}:{ip}:{port}' for port in ports])

    host = get_ssh_host(host_name)
    exe = f'ssh {host} {forwards}'
    return exe


if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
        host_name = sys.argv[2]
        subprocess.run(get_exe(get_json(file_path), host_name), shell=True)

    except:
        print(f'Usage {sys.argv[0]} [kernel-config.json] [host_name]')
