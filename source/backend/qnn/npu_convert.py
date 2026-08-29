#!/usr/bin/python
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
post_treat = {}
qnn_sdk = os.environ["QNN_SDK_ROOT"]
print(qnn_sdk)
with open(sys.argv[1]) as f:
    post_treat = json.load(f)
soc_id = int(sys.argv[2])
dsp_arch = sys.argv[3]
print('soc_id:', soc_id, "; dsp_arch:", dsp_arch)
qnn_bin_path = os.path.join(qnn_sdk, 'bin', 'x86_64-linux-clang')
qnnModelLibGenerator = os.path.join(qnn_bin_path, 'qnn-model-lib-generator')
qnnContextBinaryGenerator = os.path.join(qnn_bin_path, 'qnn-context-binary-generator')
merges = post_treat["merge"]
cache_dir = 'res'
if 'cache' in post_treat:
    cache_dir = post_treat['cache']
clean_tmp = True
separate_graphs = bool(post_treat.get("separate_graphs", False))
context_config = {
    "backend_extensions": {
        "shared_library_path": os.path.join(qnn_sdk, "lib","x86_64-linux-clang","libQnnHtpNetRunExtensions.so"),
        "config_file_path": "./htp_backend_extensions.json"
    }
}
htp_so = os.path.join(qnn_sdk, 'lib','x86_64-linux-clang','libQnnHtp.so')
with open('context_config.json', 'w') as f:
    f.write(json.dumps(context_config, indent=4))

htp_backend_extensions = {
    "graphs": [
        {
            "vtcm_mb": 8,
            "O": 3.0,
            "fp16_relaxed_precision": 1,
            "hvx_threads": 4
        }
    ],
    "devices": [
        {
            "soc_id": soc_id,
            "dsp_arch": dsp_arch,
            "cores": [
                {
                    "core_id": 0,
                    "perf_profile": "burst",
                    "rpc_control_latency": 100
                }
            ]
        }
    ],
    "context": {
        "weight_sharing_enabled": True
    }
}

for key in post_treat["merge"]:
    srcs = merges[key]
    dstname = key.split('/')
    dstname = dstname[len(dstname)-1]
    dstname = dstname.replace('.bin', '')
    graphs = []
    libs = []
    workdirs = []
    for i,src in enumerate(srcs):
        # tar
        graphname = os.path.basename(src)
        graphs.append(graphname)
        workdir = os.path.join(os.getcwd(), src)
        workdirs.append(workdir)
        raw_files = sorted(path.name for path in Path(workdir).glob('*.raw'))
        if not raw_files:
            raise RuntimeError('no raw graph files found in ' + workdir)
        subprocess.run(['tar', '-cf', graphname + '.bin', *raw_files], cwd=workdir, check=True)
        if clean_tmp:
            for raw_file in raw_files:
                (Path(workdir) / raw_file).unlink()
        # Compile
        subprocess.run([
            sys.executable,
            qnnModelLibGenerator,
            '-c', os.path.join(workdir, graphname + '.cpp'),
            '-b', os.path.join(workdir, graphname + '.bin'),
            '-t', 'x86_64-linux-clang',
            '-o', workdir,
        ], check=True)
        if clean_tmp:
            (Path(workdir) / (graphname + '.bin')).unlink()
        libs.append(os.path.join(workdir, 'x86_64-linux-clang', 'lib' + graphname + '.so'))
    if separate_graphs:
        for i, (graph, lib) in enumerate(zip(graphs, libs)):
            htp_backend_extensions['graphs'][0]['graph_names'] = [graph]
            with open('htp_backend_extensions.json', 'w') as f:
                f.write(json.dumps(htp_backend_extensions, indent=4))
            binary_name = dstname + '_' + str(i)
            subprocess.run([
                qnnContextBinaryGenerator,
                '--model', lib,
                '--backend', htp_so,
                '--binary_file', binary_name,
                '--config_file', './context_config.json',
                '--output_dir', cache_dir,
            ], check=True)
    else:
        htp_backend_extensions['graphs'][0]['graph_names'] = graphs
        with open('htp_backend_extensions.json', 'w') as f:
            f.write(json.dumps(htp_backend_extensions, indent=4))
        subprocess.run([
            qnnContextBinaryGenerator,
            '--model', ','.join(libs),
            '--backend', htp_so,
            '--binary_file', dstname,
            '--config_file', './context_config.json',
            '--output_dir', cache_dir,
        ], check=True)
    if clean_tmp:
        for workdir in workdirs:
            shutil.rmtree(workdir)
